package vm

//go:generate sh -c "go run ./func_types > ./func_types[generated].go"

import (
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/expr-lang/expr/builtin"
	"github.com/expr-lang/expr/conf"
	"github.com/expr-lang/expr/file"
	"github.com/expr-lang/expr/internal/deref"
	"github.com/expr-lang/expr/vm/runtime"
)

const maxFnArgsBuf = 256

func Run(program *Program, env any) (any, error) {
	if program == nil {
		return nil, fmt.Errorf("program is nil")
	}
	vm := VM{}
	return vm.Run(program, env)
}

func Debug() *VM {
	vm := &VM{
		debug: true,
		step:  make(chan struct{}, 0),
		curr:  make(chan int, 0),
	}
	return vm
}

type VM struct {
	Stack        []any
	Scopes       []*Scope
	Variables    []any
	MemoryBudget uint
	ip           int
	memory       uint
	debug        bool
	step         chan struct{}
	curr         chan int
	scopePool    []Scope // Pre-allocated pool of Scope values; grows as needed but never shrinks
	scopePoolIdx int     // Current index into scopePool for allocation
	currScope    *Scope  // Cached pointer to the current scope (optimization)
	tryFrames    []tryFrame
	fnArgsBuf    []any // Per-run reusable buffer for function call arguments
}

// tryFrame is a snapshot of VM state taken by OpTryBegin. When evaluation
// of the primary expression fails, the VM rolls back to this snapshot
// before evaluating the fallback expression.
type tryFrame struct {
	stackLen   int
	scopesLen  int
	poolIdx    int
	currScope  *Scope
	memory     uint
	fallbackIP int
}

// errMemoryBudgetExceeded is panicked by memGrow when the memory budget is
// exhausted. Unlike ordinary runtime evaluation errors, it must never be
// caught by a try() primary expression.
var errMemoryBudgetExceeded = errors.New("memory budget exceeded")

func (vm *VM) Run(program *Program, env any) (_ any, err error) {
	defer func() {
		if r := recover(); r != nil {
			var location file.Location
			if vm.ip-1 < len(program.locations) {
				location = program.locations[vm.ip-1]
			}
			f := &file.Error{
				Location: location,
				Message:  fmt.Sprintf("%v", r),
			}
			if err, ok := r.(error); ok {
				f.Wrap(err)
			}
			err = f.Bind(program.source)
		}
	}()

	if vm.Stack == nil {
		vm.Stack = make([]any, 0, 2)
	} else {
		clearSlice(vm.Stack)
		vm.Stack = vm.Stack[0:0]
	}
	if vm.Scopes != nil {
		clearSlice(vm.Scopes)
		vm.Scopes = vm.Scopes[0:0]
	}
	vm.scopePoolIdx = 0 // Reset pool index for reuse
	vm.currScope = nil
	if len(vm.Variables) < program.variables {
		vm.Variables = make([]any, program.variables)
	}
	if vm.MemoryBudget == 0 {
		vm.MemoryBudget = conf.DefaultMemoryBudget
	}
	vm.memory = 0
	vm.ip = 0
	vm.fnArgsBuf = nil
	clearSlice(vm.tryFrames)
	vm.tryFrames = vm.tryFrames[:0]

	for vm.ip < len(program.Bytecode) {
		if debug && vm.debug {
			<-vm.step
		}

		vm.runOpcode(program, env)

		if debug && vm.debug {
			vm.curr <- vm.ip
		}
	}

	if debug && vm.debug {
		close(vm.curr)
		close(vm.step)
	}

	if len(vm.Stack) > 0 {
		return vm.pop(), nil
	}

	return nil, nil
}

// runOpcode executes exactly one bytecode instruction. A try() primary
// expression is driven by runProtected instead of the main loop, so that a
// panic inside it can be recovered without unwinding the whole Run.
func (vm *VM) runOpcode(program *Program, env any) {
	op := program.Bytecode[vm.ip]
	arg := program.Arguments[vm.ip]
	vm.ip += 1

	switch op {

	case OpInvalid:
		panic("invalid opcode")

	case OpPush:
		vm.push(program.Constants[arg])

	case OpInt:
		vm.push(arg)

	case OpPop:
		vm.pop()

	case OpStore:
		vm.Variables[arg] = vm.pop()

	case OpLoadVar:
		vm.push(vm.Variables[arg])

	case OpLoadConst:
		vm.push(runtime.Fetch(env, program.Constants[arg]))

	case OpLoadField:
		vm.push(runtime.FetchField(env, program.Constants[arg].(*runtime.Field)))

	case OpLoadFast:
		vm.push(env.(map[string]any)[program.Constants[arg].(string)])

	case OpLoadMethod:
		vm.push(runtime.FetchMethod(env, program.Constants[arg].(*runtime.Method)))

	case OpLoadFunc:
		vm.push(program.functions[arg])

	case OpFetch:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Fetch(a, b))

	case OpFetchField:
		a := vm.pop()
		vm.push(runtime.FetchField(a, program.Constants[arg].(*runtime.Field)))

	case OpLoadEnv:
		vm.push(env)

	case OpMethod:
		a := vm.pop()
		vm.push(runtime.FetchMethod(a, program.Constants[arg].(*runtime.Method)))

	case OpTrue:
		vm.push(true)

	case OpFalse:
		vm.push(false)

	case OpNil:
		vm.push(nil)

	case OpNegate:
		v := runtime.Negate(vm.pop())
		vm.push(v)

	case OpNot:
		v := vm.pop().(bool)
		vm.push(!v)

	case OpEqual:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Equal(a, b))

	case OpEqualInt:
		b := vm.pop()
		a := vm.pop()
		vm.push(a.(int) == b.(int))

	case OpEqualString:
		b := vm.pop()
		a := vm.pop()
		vm.push(a.(string) == b.(string))

	case OpJump:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		vm.ip += arg

	case OpJumpIfTrue:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		if vm.current().(bool) {
			vm.ip += arg
		}

	case OpJumpIfFalse:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		if !vm.current().(bool) {
			vm.ip += arg
		}

	case OpJumpIfNil:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		if runtime.IsNil(vm.current()) {
			vm.ip += arg
		}

	case OpJumpIfNotNil:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		if !runtime.IsNil(vm.current()) {
			vm.ip += arg
		}

	case OpJumpIfEnd:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		if vm.currScope.Index >= vm.currScope.Len {
			vm.ip += arg
		}

	case OpJumpBackward:
		vm.ip -= arg

	case OpIn:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.In(a, b))

	case OpLess:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Less(a, b))

	case OpMore:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.More(a, b))

	case OpLessOrEqual:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.LessOrEqual(a, b))

	case OpMoreOrEqual:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.MoreOrEqual(a, b))

	case OpAdd:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Add(a, b))

	case OpSubtract:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Subtract(a, b))

	case OpMultiply:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Multiply(a, b))

	case OpDivide:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Divide(a, b))

	case OpModulo:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Modulo(a, b))

	case OpExponent:
		b := vm.pop()
		a := vm.pop()
		vm.push(runtime.Exponent(a, b))

	case OpRange:
		b := vm.pop()
		a := vm.pop()
		min := runtime.ToInt(a)
		max := runtime.ToInt(b)
		size := max - min + 1
		if size <= 0 {
			size = 0
		}
		vm.memGrow(uint(size))
		vm.push(runtime.MakeRange(min, max))

	case OpMatches:
		b := vm.pop()
		a := vm.pop()
		if runtime.IsNil(a) || runtime.IsNil(b) {
			vm.push(false)
			break
		}
		var match bool
		var err error
		if s, ok := a.(string); ok {
			match, err = regexp.MatchString(b.(string), s)
		} else {
			match, err = regexp.Match(b.(string), a.([]byte))
		}
		if err != nil {
			panic(err)
		}
		vm.push(match)

	case OpMatchesConst:
		a := vm.pop()
		if runtime.IsNil(a) {
			vm.push(false)
			break
		}
		r := program.Constants[arg].(*regexp.Regexp)
		if s, ok := a.(string); ok {
			vm.push(r.MatchString(s))
		} else {
			vm.push(r.Match(a.([]byte)))
		}

	case OpContains:
		b := vm.pop()
		a := vm.pop()
		if runtime.IsNil(a) || runtime.IsNil(b) {
			vm.push(false)
			break
		}
		vm.push(strings.Contains(a.(string), b.(string)))

	case OpStartsWith:
		b := vm.pop()
		a := vm.pop()
		if runtime.IsNil(a) || runtime.IsNil(b) {
			vm.push(false)
			break
		}
		vm.push(strings.HasPrefix(a.(string), b.(string)))

	case OpEndsWith:
		b := vm.pop()
		a := vm.pop()
		if runtime.IsNil(a) || runtime.IsNil(b) {
			vm.push(false)
			break
		}
		vm.push(strings.HasSuffix(a.(string), b.(string)))

	case OpSlice:
		from := vm.pop()
		to := vm.pop()
		node := vm.pop()
		vm.push(runtime.Slice(node, from, to))

	case OpCall:
		v := vm.pop()
		if v == nil {
			panic("invalid operation: cannot call nil")
		}
		fn := reflect.ValueOf(v)
		if fn.Kind() != reflect.Func {
			panic(fmt.Sprintf("invalid operation: cannot call non-function of type %T", v))
		}
		fnType := fn.Type()
		size := arg
		isVariadic := fnType.IsVariadic()
		numIn := fnType.NumIn()
		if isVariadic {
			if size < numIn-1 {
				panic(fmt.Sprintf("invalid number of arguments: expected at least %d, got %d", numIn-1, size))
			}
		} else {
			if size != numIn {
				panic(fmt.Sprintf("invalid number of arguments: expected %d, got %d", numIn, size))
			}
		}
		in := make([]reflect.Value, size)
		for i := int(size) - 1; i >= 0; i-- {
			param := vm.pop()
			if param == nil {
				var inType reflect.Type
				if isVariadic && i >= numIn-1 {
					inType = fnType.In(numIn - 1).Elem()
				} else {
					inType = fnType.In(i)
				}
				in[i] = reflect.Zero(inType)
			} else {
				in[i] = reflect.ValueOf(param)
			}
		}
		out := fn.Call(in)
		if len(out) == 2 && out[1].Type() == errorType && !out[1].IsNil() {
			panic(out[1].Interface().(error))
		}
		vm.push(out[0].Interface())

	case OpCall0:
		out, err := program.functions[arg]()
		if err != nil {
			panic(err)
		}
		vm.push(out)

	case OpCall1:
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, 1)
		out, err := program.functions[arg](args...)
		if err != nil {
			panic(err)
		}
		vm.push(out)

	case OpCall2:
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, 2)
		out, err := program.functions[arg](args...)
		if err != nil {
			panic(err)
		}
		vm.push(out)

	case OpCall3:
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, 3)
		out, err := program.functions[arg](args...)
		if err != nil {
			panic(err)
		}
		vm.push(out)

	case OpCallN:
		fn := vm.pop().(Function)
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, arg)
		out, err := fn(args...)
		if err != nil {
			panic(err)
		}
		vm.push(out)

	case OpCallFast:
		fn := vm.pop().(func(...any) any)
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, arg)
		vm.push(fn(args...))

	case OpCallSafe:
		fn := vm.pop().(SafeFunction)
		var args []any
		args, vm.fnArgsBuf = vm.getArgsForFunc(vm.fnArgsBuf, program, arg)
		out, mem, err := fn(args...)
		if err != nil {
			panic(err)
		}
		vm.memGrow(mem)
		vm.push(out)

	case OpCallTyped:
		vm.push(vm.call(vm.pop(), arg))

	case OpCallBuiltin1:
		vm.push(builtin.Builtins[arg].Fast(vm.pop()))

	case OpArray:
		size := vm.pop().(int)
		vm.memGrow(uint(size))
		array := make([]any, size)
		for i := size - 1; i >= 0; i-- {
			array[i] = vm.pop()
		}
		vm.push(array)

	case OpMap:
		size := vm.pop().(int)
		vm.memGrow(uint(size))
		m := make(map[string]any)
		for i := size - 1; i >= 0; i-- {
			value := vm.pop()
			key := vm.pop()
			m[key.(string)] = value
		}
		vm.push(m)

	case OpLen:
		vm.push(runtime.Len(vm.current()))

	case OpCast:
		switch arg {
		case 0:
			vm.push(runtime.ToInt(vm.pop()))
		case 1:
			vm.push(runtime.ToInt64(vm.pop()))
		case 2:
			vm.push(runtime.ToFloat64(vm.pop()))
		case 3:
			vm.push(runtime.ToBool(vm.pop()))
		}

	case OpDeref:
		a := vm.pop()
		vm.push(deref.Interface(a))

	case OpIncrementIndex:
		vm.currScope.Index++

	case OpDecrementIndex:
		vm.currScope.Index--

	case OpIncrementCount:
		vm.currScope.Count++

	case OpGetIndex:
		vm.push(vm.currScope.Index)

	case OpGetCount:
		vm.push(vm.currScope.Count)

	case OpGetLen:
		vm.push(vm.currScope.Len)

	case OpGetAcc:
		vm.push(vm.currScope.Acc)

	case OpSetAcc:
		vm.currScope.Acc = vm.pop()

	case OpSetIndex:
		vm.currScope.Index = vm.pop().(int)

	case OpPointer:
		vm.push(vm.currScope.Item())

	case OpThrow:
		panic(vm.pop().(error))

	case OpCreate:
		switch arg {
		case 1:
			vm.push(make(groupBy))
		case 2:
			scope := vm.currScope
			var desc bool
			order, ok := vm.pop().(string)
			if !ok {
				panic("sortBy order argument must be a string")
			}
			switch order {
			case "asc":
				desc = false
			case "desc":
				desc = true
			default:
				panic("unknown order, use asc or desc")
			}
			vm.push(&runtime.SortBy{
				Desc:   desc,
				Array:  make([]any, 0, scope.Len),
				Values: make([]any, 0, scope.Len),
			})
		default:
			panic(fmt.Sprintf("unknown OpCreate argument %v", arg))
		}

	case OpGroupBy:
		scope := vm.currScope
		key := vm.pop()
		if key != nil && !reflect.TypeOf(key).Comparable() {
			panic(fmt.Sprintf("cannot use %T as a key for groupBy: type is not comparable", key))
		}
		scope.Acc.(groupBy)[key] = append(scope.Acc.(groupBy)[key], scope.Item())

	case OpSortBy:
		scope := vm.currScope
		value := vm.pop()
		sortable := scope.Acc.(*runtime.SortBy)
		sortable.Array = append(sortable.Array, scope.Item())
		sortable.Values = append(sortable.Values, value)

	case OpSort:
		scope := vm.currScope
		sortable := scope.Acc.(*runtime.SortBy)
		sort.Sort(sortable)
		vm.memGrow(uint(scope.Len))
		vm.push(sortable.Array)

	case OpProfileStart:
		span := program.Constants[arg].(*Span)
		span.start = time.Now()

	case OpProfileEnd:
		span := program.Constants[arg].(*Span)
		span.Duration += time.Since(span.start).Nanoseconds()

	case OpBegin:
		a := vm.pop()
		s := vm.allocScope()
		switch v := a.(type) {
		case []int:
			s.Ints = v
			s.Len = len(v)
		case []float64:
			s.Floats = v
			s.Len = len(v)
		case []string:
			s.Strings = v
			s.Len = len(v)
		case []any:
			s.Anys = v
			s.Len = len(v)
		default:
			s.Array = reflect.ValueOf(a)
			s.Len = s.Array.Len()
		}
		vm.Scopes = append(vm.Scopes, s)
		vm.currScope = s

	case OpAnd:
		a := vm.pop()
		b := vm.pop()
		vm.push(a.(bool) && b.(bool))

	case OpOr:
		a := vm.pop()
		b := vm.pop()
		vm.push(a.(bool) || b.(bool))

	case OpTryBegin:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		depth := len(vm.tryFrames)
		vm.tryFrames = append(vm.tryFrames, tryFrame{
			stackLen:   len(vm.Stack),
			scopesLen:  len(vm.Scopes),
			poolIdx:    vm.scopePoolIdx,
			currScope:  vm.currScope,
			memory:     vm.memory,
			fallbackIP: vm.ip + arg,
		})
		vm.runProtected(program, env, depth)

	case OpTryJump:
		if arg < 0 {
			panic("negative jump offset is invalid")
		}
		// The primary expression succeeded: leave its result on the
		// stack, drop the protection frame and skip the fallback.
		vm.tryFrames = vm.tryFrames[:len(vm.tryFrames)-1]
		vm.ip += arg

	case OpEnd:
		vm.Scopes = vm.Scopes[:len(vm.Scopes)-1]
		if len(vm.Scopes) > 0 {
			vm.currScope = vm.Scopes[len(vm.Scopes)-1]
		} else {
			vm.currScope = nil
		}

	default:
		panic(fmt.Sprintf("unknown bytecode %#x", op))
	}
}

func (vm *VM) push(value any) {
	vm.Stack = append(vm.Stack, value)
}

func (vm *VM) current() any {
	if len(vm.Stack) == 0 {
		panic("stack underflow")
	}
	return vm.Stack[len(vm.Stack)-1]
}

func (vm *VM) pop() any {
	if len(vm.Stack) == 0 {
		panic("stack underflow")
	}
	value := vm.Stack[len(vm.Stack)-1]
	vm.Stack = vm.Stack[:len(vm.Stack)-1]
	return value
}

func (vm *VM) memGrow(size uint) {
	vm.memory += size
	if vm.memory >= vm.MemoryBudget {
		panic(errMemoryBudgetExceeded)
	}
}

// runProtected drives the evaluation of a try() primary expression, which
// is delimited by the try frame at index depth and its OpTryJump
// instruction. If the primary panics with an ordinary runtime evaluation
// error, the VM state is rolled back to the frame snapshot and execution
// continues at the fallback expression; a memory budget error is
// re-panicked so that it still aborts the whole program.
func (vm *VM) runProtected(program *Program, env any, depth int) {
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok && errors.Is(err, errMemoryBudgetExceeded) {
				panic(r)
			}
			frame := vm.tryFrames[depth]

			// Drop values left on the stack by the failed primary.
			if len(vm.Stack) > frame.stackLen {
				clearSlice(vm.Stack[frame.stackLen:])
				vm.Stack = vm.Stack[:frame.stackLen]
			}

			// Close collection scopes left open by the failed primary.
			if len(vm.Scopes) > frame.scopesLen {
				clearSlice(vm.Scopes[frame.scopesLen:])
				vm.Scopes = vm.Scopes[:frame.scopesLen]
			}
			vm.currScope = frame.currScope
			vm.scopePoolIdx = frame.poolIdx

			// Values produced by the failed primary are discarded, so
			// their accounted memory is released too.
			vm.memory = frame.memory

			// The fallback runs without this protection frame: its own
			// failure must propagate with the fallback's error location.
			clearSlice(vm.tryFrames[depth:])
			vm.tryFrames = vm.tryFrames[:depth]
			vm.ip = frame.fallbackIP
		}
	}()

	for len(vm.tryFrames) > depth {
		if debug && vm.debug {
			<-vm.step
		}

		vm.runOpcode(program, env)

		if debug && vm.debug {
			vm.curr <- vm.ip
		}
	}
}

func (vm *VM) scope() *Scope {
	return vm.Scopes[len(vm.Scopes)-1]
}

// allocScope returns a pointer to a Scope from the pool, growing the pool if needed.
// Callers must set Len and exactly one of: Ints, Floats, Strings, Anys, or Array.
func (vm *VM) allocScope() *Scope {
	if vm.scopePoolIdx >= len(vm.scopePool) {
		vm.scopePool = append(vm.scopePool, Scope{})
	}
	s := &vm.scopePool[vm.scopePoolIdx]
	vm.scopePoolIdx++
	// Reset iteration state
	s.Index = 0
	s.Count = 0
	s.Acc = nil
	// Clear typed slice pointers to avoid stale fast-path matches
	s.Ints = nil
	s.Floats = nil
	s.Strings = nil
	s.Anys = nil
	// Clear Array to release reference for GC (only matters for fallback path)
	s.Array = reflect.Value{}
	return s
}

// getArgsForFunc lazily initializes the buffer the first time it is called for
// a given program (thus, it also needs "program" to run). It will
// take "needed" elements from the buffer and populate them with vm.pop() in
// reverse order. Because the estimation can fall short, this function can
// occasionally make a new allocation.
func (vm *VM) getArgsForFunc(argsBuf []any, program *Program, needed int) (args []any, argsBufOut []any) {
	if needed == 0 || program == nil {
		return nil, argsBuf
	}

	// Step 1: fix estimations and preallocate
	if argsBuf == nil {
		estimatedFnArgsCount := estimateFnArgsCount(program)
		if estimatedFnArgsCount > maxFnArgsBuf {
			// put a practical limit to avoid excessive preallocation
			estimatedFnArgsCount = maxFnArgsBuf
		}
		if estimatedFnArgsCount < needed {
			// in the case that the first call is for example OpCallN with a large
			// number of arguments, then make sure we will be able to serve them at
			// least.
			estimatedFnArgsCount = needed
		}

		// in the case that we are preparing the arguments for the first
		// function call of the program, then argsBuf will be nil, so we
		// initialize it. We delay this initial allocation here because a
		// program could have many function calls but exit earlier than the
		// first call, so in that case we avoid allocating unnecessarily
		argsBuf = make([]any, estimatedFnArgsCount)
	}

	// Step 2: get the final slice that will be returned
	var buf []any
	if len(argsBuf) >= needed {
		// in this case, we are successfully using the single preallocation. We
		// use the full slice expression [low : high : max] because in that way
		// a function that receives this slice as variadic arguments will not be
		// able to make modifications to contiguous elements with append(). If
		// they call append on their variadic arguments they will make a new
		// allocation.
		buf = (argsBuf)[:needed:needed]
		argsBuf = (argsBuf)[needed:] // advance the buffer
	} else {
		// if we have been making calls to something like OpCallN with many more
		// arguments than what we estimated, then we will need to allocate
		// separately
		buf = make([]any, needed)
	}

	// Step 3: populate the final slice bulk copying from the stack. This is the
	// exact order and copy() is a highly optimized operation
	copy(buf, vm.Stack[len(vm.Stack)-needed:])
	vm.Stack = vm.Stack[:len(vm.Stack)-needed]

	return buf, argsBuf
}

func (vm *VM) Step() {
	vm.step <- struct{}{}
}

func (vm *VM) Position() chan int {
	return vm.curr
}

func clearSlice[S ~[]E, E any](s S) {
	var zero E
	for i := range s {
		s[i] = zero // clear mem, optimized by the compiler, in Go 1.21 the "clear" builtin can be used
	}
}

// estimateFnArgsCount inspects a *Program and estimates how many function
// arguments will be required to run it.
func estimateFnArgsCount(program *Program) int {
	// Implementation note: a program will not necessarily go through all
	// operations, but this is just an estimation
	var count int
	for _, op := range program.Bytecode {
		if int(op) < len(opArgLenEstimation) {
			count += opArgLenEstimation[op]
		}
	}
	return count
}

var opArgLenEstimation = [...]int{
	OpCall1: 1,
	OpCall2: 2,
	OpCall3: 3,
	// we don't know exactly but we know at least 4, so be conservative as this
	// is only an optimization and we also want to avoid excessive preallocation
	OpCallN: 4,
	// here we don't know either, but we can guess it could be common to receive
	// up to 3 arguments in a function
	OpCallFast: 3,
	OpCallSafe: 3,
}

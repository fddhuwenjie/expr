package vm_test

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/expr-lang/expr/file"
	"github.com/expr-lang/expr/internal/testify/require"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"
)

// eventCollector 是线程安全的事件收集器。
type eventCollector struct {
	mu     sync.Mutex
	events []vm.Event
}

func (c *eventCollector) observe(event vm.Event) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, event)
	return nil
}

func (c *eventCollector) snapshot() []vm.Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]vm.Event, len(c.events))
	copy(out, c.events)
	return out
}

// requireContinuousSeq 校验事件序号在单次运行内从 1 开始连续递增。
func requireContinuousSeq(t *testing.T, events []vm.Event) {
	t.Helper()
	for i, event := range events {
		require.Equalf(t, uint64(i+1), event.Seq, "第 %d 个事件的序号应为 %d，实际为 %d", i, i+1, event.Seq)
	}
}

// eventWithLocation 返回指定指令、源位置区间恰好为 [from, to) 的第一个事件。
func eventWithLocation(t *testing.T, events []vm.Event, opcode vm.Opcode, from, to int) vm.Event {
	t.Helper()
	for _, event := range events {
		if event.Opcode == opcode && event.Location.From == from && event.Location.To == to {
			return event
		}
	}
	t.Fatalf("未找到指令 %v 源位置 [%d,%d) 的事件，实际事件位置：%v", opcode, from, to, locationsOf(events))
	return vm.Event{}
}

// ipsInSourceRange 静态地收集源位置完全落在 fragment 区间内的字节码 IP。
func ipsInSourceRange(t *testing.T, program *vm.Program, source, fragment string) []int {
	t.Helper()
	from := strings.Index(source, fragment)
	require.GreaterOrEqualf(t, from, 0, "片段 %q 不在源码 %q 中", fragment, source)
	to := from + len(fragment)
	var ips []int
	for ip, location := range program.Locations() {
		if location.From >= from && location.To <= to && location.To > location.From {
			ips = append(ips, ip)
		}
	}
	require.NotEmpty(t, ips)
	return ips
}

func executedIPs(events []vm.Event) map[int]bool {
	executed := make(map[int]bool, len(events))
	for _, event := range events {
		executed[event.IP] = true
	}
	return executed
}

func locationsOf(events []vm.Event) []file.Location {
	out := make([]file.Location, len(events))
	for i, event := range events {
		out[i] = event.Location
	}
	return out
}

// TestObserver_Success 校验成功执行时事件按真实执行顺序交付，且事件字段与
// 字节码、源位置和栈状态一致。
func TestObserver_Success(t *testing.T) {
	program, err := expr.Compile(`a + b`,
		expr.Env(map[string]any{"a": 0, "b": 0}),
		expr.Optimize(false),
	)
	require.NoError(t, err)

	collector := &eventCollector{}
	out, err := vm.RunObserve(program, map[string]any{"a": 1, "b": 2}, collector.observe)
	require.NoError(t, err)
	require.Equal(t, 3, out)

	events := collector.snapshot()
	requireContinuousSeq(t, events)

	// 无跳转的直线程序中，每条字节码指令都应恰好执行一次。
	require.Len(t, events, len(program.Bytecode))

	runID := events[0].RunID
	require.NotZero(t, runID)
	locations := program.Locations()
	for i, event := range events {
		require.Equalf(t, uint64(i+1), event.Seq, "事件序号必须连续")
		require.Equalf(t, runID, event.RunID, "同一运行的 RunID 必须一致")
		require.Equalf(t, 0, event.Layer, "顶层程序事件必须位于调用层 0")
		require.Equalf(t, i, event.IP, "直线程序的第 %d 个事件 IP 应为 %d", i, i)
		require.Equalf(t, program.Bytecode[i], event.Opcode, "事件指令必须与字节码一致")
		require.Equalf(t, program.Arguments[i], event.Argument, "事件参数必须与字节码参数一致")
		require.Equalf(t, locations[i], event.Location, "事件源位置必须与字节码位置表一致")
		require.GreaterOrEqualf(t, event.StackDepth, 0, "栈深度不能为负")
	}
	// 程序结束时栈上只保留最终结果。
	require.Equal(t, 1, events[len(events)-1].StackDepth)

	// 再次运行必须得到完全一致的事件流（程序可复用、结果确定）；仅 RunID
	// 因属于不同运行而不同，归一化后两个事件流必须逐条相等。
	again := &eventCollector{}
	out2, err := vm.RunObserve(program, map[string]any{"a": 40, "b": 2}, again.observe)
	require.NoError(t, err)
	require.Equal(t, 42, out2)
	againEvents := again.snapshot()
	require.NotEqual(t, runID, againEvents[0].RunID)
	for i := range againEvents {
		againEvents[i].RunID = runID
	}
	require.Equal(t, events, againEvents)
}

// TestObserver_EventImmutable 校验事件本身是不可变的值对象：所有字段都是
// 值类型，观察函数持有的事件不会关联虚拟机内部可变状态。
func TestObserver_EventImmutable(t *testing.T) {
	eventType := reflect.TypeOf(vm.Event{})
	for i := 0; i < eventType.NumField(); i++ {
		field := eventType.Field(i)
		switch field.Type.Kind() {
		case reflect.Interface, reflect.Map, reflect.Slice, reflect.Ptr, reflect.Chan, reflect.Func, reflect.UnsafePointer:
			t.Fatalf("Event.%s 是引用类型 %s，事件必须是不可变值对象", field.Name, field.Type)
		}
	}

	program, err := expr.Compile(`a + b`, expr.Env(map[string]any{"a": 0, "b": 0}), expr.Optimize(false))
	require.NoError(t, err)

	// 观察函数内修改收到的事件副本，不得影响后续事件。
	var first vm.Event
	var touched int32
	observer := func(event vm.Event) error {
		if atomic.AddInt32(&touched, 1) == 1 {
			first = event
			event.Seq = 999
			event.StackDepth = 999
			event.RunID = 999
		}
		return nil
	}
	_, err = vm.RunObserve(program, map[string]any{"a": 1, "b": 2}, observer)
	require.NoError(t, err)
	require.Equal(t, uint64(1), first.Seq)
}

// TestObserver_ShortCircuit 校验短路与非短路执行的事件边界：被短路的操作数
// 指令一条也不能产生事件。
func TestObserver_ShortCircuit(t *testing.T) {
	var mu sync.Mutex
	calls := map[string]int{}
	call := func(name string) func() bool {
		return func() bool {
			mu.Lock()
			calls[name]++
			mu.Unlock()
			return true
		}
	}
	env := map[string]any{
		"foo": call("foo"),
		"bar": call("bar"),
		"baz": call("baz"),
	}

	// false and bar()：bar() 必须被短路。
	program, err := expr.Compile(`foo() and bar()`,
		expr.Env(env),
		// foo 返回 false。
	)
	require.NoError(t, err)
	env["foo"] = func() bool {
		mu.Lock()
		calls["foo"]++
		mu.Unlock()
		return false
	}
	collector := &eventCollector{}
	out, err := vm.RunObserve(program, env, collector.observe)
	require.NoError(t, err)
	require.Equal(t, false, out)

	andSource := `foo() and bar()`
	events := collector.snapshot()
	requireContinuousSeq(t, events)
	executed := executedIPs(events)
	for _, ip := range ipsInSourceRange(t, program, andSource, "bar()") {
		require.Falsef(t, executed[ip], "短路时 bar 区域内的指令 IP=%d 不得执行", ip)
	}
	require.Equal(t, vm.OpJumpIfFalse, events[len(events)-1].Opcode, "短路跳转必须是最后一条真实执行的指令")

	mu.Lock()
	require.Equal(t, 0, calls["bar"], "短路时 bar 不得被执行")
	mu.Unlock()

	// 关闭短路后，bar 必须执行并产生事件。
	programNoShort, err := expr.Compile(andSource, expr.Env(env), expr.DisableShortCircuit())
	require.NoError(t, err)
	collector2 := &eventCollector{}
	env["foo"] = call("foo")
	out, err = vm.RunObserve(programNoShort, env, collector2.observe)
	require.NoError(t, err)
	require.Equal(t, true, out)
	events2 := collector2.snapshot()
	executed2 := executedIPs(events2)
	barExecuted := false
	for _, ip := range ipsInSourceRange(t, programNoShort, andSource, "bar()") {
		if executed2[ip] {
			barExecuted = true
		}
	}
	require.True(t, barExecuted, "关闭短路后 bar 区域必须执行")
	mu.Lock()
	require.Equal(t, 1, calls["bar"])
	mu.Unlock()

	// true or baz()：or 的右侧同样必须被短路。
	orSource := `foo() or baz()`
	progOr, err := expr.Compile(orSource, expr.Env(env))
	require.NoError(t, err)
	collector3 := &eventCollector{}
	out, err = vm.RunObserve(progOr, env, collector3.observe)
	require.NoError(t, err)
	require.Equal(t, true, out)
	events3 := collector3.snapshot()
	executed3 := executedIPs(events3)
	for _, ip := range ipsInSourceRange(t, progOr, orSource, "baz()") {
		require.Falsef(t, executed3[ip], "短路时 baz 区域内的指令 IP=%d 不得执行", ip)
	}
	mu.Lock()
	bazCalls := calls["baz"]
	mu.Unlock()
	require.Equal(t, 0, bazCalls)
}

// TestObserver_SafeCall 校验安全调用链在命中 nil 时，被跳过的链路段不产生
// 事件；非 nil 时链路指令正常执行。
func TestObserver_SafeCall(t *testing.T) {
	source := `foo?.bar`
	// 不指定 Env 类型，foo 在运行时以 map 形式提供；bar 的读取走 OpFetch。
	program, err := expr.Compile(source, expr.AllowUndefinedVariables())
	require.NoError(t, err)

	// nil 链路：OpJumpIfNil 之后到跳转目标之间的指令全部跳过。
	collector := &eventCollector{}
	out, err := vm.RunObserve(program, map[string]any{"foo": nil}, collector.observe)
	require.NoError(t, err)
	require.Nil(t, out)

	events := collector.snapshot()
	requireContinuousSeq(t, events)

	var jump vm.Event
	for _, event := range events {
		if event.Opcode == vm.OpJumpIfNil {
			jump = event
		}
	}
	require.Equal(t, vm.OpJumpIfNil, jump.Opcode, "安全调用必须包含 OpJumpIfNil")
	target := jump.IP + 1 + jump.Argument
	executed := executedIPs(events)
	for ip := jump.IP + 1; ip < target; ip++ {
		require.Falsef(t, executed[ip], "命中 nil 的安全调用不得执行跳转区间内的指令 IP=%d", ip)
	}

	// 非 nil 链路：被保护的 bar 读取（OpFetch）必须执行。
	collector2 := &eventCollector{}
	out, err = vm.RunObserve(program, map[string]any{"foo": map[string]any{"bar": 42}}, collector2.observe)
	require.NoError(t, err)
	require.Equal(t, 42, out)
	events2 := collector2.snapshot()
	requireContinuousSeq(t, events2)
	executed2 := executedIPs(events2)
	require.True(t, executed2[jump.IP])
	require.True(t, executed2[target-1], "非 nil 时被保护的链路段必须得到执行")
}

// TestObserver_NestedFunctions 校验嵌套求值（内联谓词、Go 函数嵌套调用以及
// 通过 RunNested 进入的子程序）的事件顺序与调用层标识。
func TestObserver_NestedFunctions(t *testing.T) {
	t.Run("内联谓词嵌套", func(t *testing.T) {
		source := `map(filter(1..6, # > 2), # * 2)`
		program, err := expr.Compile(source, expr.Optimize(false))
		require.NoError(t, err)

		collector := &eventCollector{}
		out, err := vm.RunObserve(program, nil, collector.observe)
		require.NoError(t, err)
		require.Equal(t, []any{6, 8, 10, 12}, out)

		events := collector.snapshot()
		// 循环回跳使同一指令被真实执行多次，事件数应大于字节码条数。
		require.Greater(t, len(events), len(program.Bytecode))
		requireContinuousSeq(t, events)
		for _, event := range events {
			require.Equal(t, 0, event.Layer)
			require.Equal(t, events[0].RunID, event.RunID)
		}
	})

	t.Run("Go 函数嵌套调用的顺序", func(t *testing.T) {
		source := `double(add(1, 2))`
		var mu sync.Mutex
		order := []string{}
		env := map[string]any{
			"add": func(a, b int) int {
				mu.Lock()
				order = append(order, "add")
				mu.Unlock()
				return a + b
			},
			"double": func(a int) int {
				mu.Lock()
				order = append(order, "double")
				mu.Unlock()
				return a * 2
			},
		}
		program, err := expr.Compile(source, expr.Env(env), expr.Optimize(false))
		require.NoError(t, err)

		collector := &eventCollector{}
		out, err := vm.RunObserve(program, env, collector.observe)
		require.NoError(t, err)
		require.Equal(t, 6, out)

		events := collector.snapshot()
		requireContinuousSeq(t, events)
		// CallNode 的源位置只覆盖被调用函数名：add 位于 [7,10)，double 位于 [0,6)。
		addIdx := strings.Index(source, "add")
		doubleIdx := strings.Index(source, "double")
		addCall := eventWithLocation(t, events, vm.OpCallTyped, addIdx, addIdx+len("add"))
		doubleCall := eventWithLocation(t, events, vm.OpCallTyped, doubleIdx, doubleIdx+len("double"))
		require.Equal(t, vm.OpCallTyped, addCall.Opcode)
		require.Equal(t, vm.OpCallTyped, doubleCall.Opcode)
		require.Less(t, addCall.Seq, doubleCall.Seq, "内层调用必须先于外层调用完成")
		mu.Lock()
		require.Equal(t, []string{"add", "double"}, order)
		mu.Unlock()
	})

	t.Run("RunNested 调用层隔离与连续序号", func(t *testing.T) {
		subProgram, err := expr.Compile(`x + 10`,
			expr.Env(map[string]any{"x": 0}),
			expr.Optimize(false),
		)
		require.NoError(t, err)

		parentVM := &vm.VM{}
		fn := expr.Function("compute", func(params ...any) (any, error) {
			return parentVM.RunNested(subProgram, map[string]any{"x": params[0]})
		}, func(int) any { return nil })
		source := `[compute(1), compute(20)]`
		program, err := expr.Compile(source,
			expr.Env(map[string]any{}),
			expr.Optimize(false),
			fn,
		)
		require.NoError(t, err)

		collector := &eventCollector{}
		out, err := parentVM.RunObserve(program, map[string]any{}, collector.observe)
		require.NoError(t, err)
		require.Equal(t, []any{11, 30}, out)

		events := collector.snapshot()
		requireContinuousSeq(t, events)

		runID := events[0].RunID
		var layers []int
		for _, event := range events {
			require.Equal(t, runID, event.RunID, "嵌套运行必须共享同一个 RunID")
			if len(layers) == 0 || layers[len(layers)-1] != event.Layer {
				layers = append(layers, event.Layer)
			}
			if event.Layer == 1 {
				// 子程序事件的位置必须落在子程序源码范围内。
				require.LessOrEqual(t, event.Location.To, len(`x + 10`))
			}
		}
		// 期望的层次序列：进入第一层子程序 -> 返回顶层 -> 进入第二层 -> 返回顶层。
		require.Equal(t, []int{0, 1, 0, 1, 0}, layers, "事件边界：%v", events)

		// 不使用观察入口时 RunNested 退化为普通执行。
		plainVM := &vm.VM{}
		plainFn := expr.Function("compute", func(params ...any) (any, error) {
			return plainVM.RunNested(subProgram, map[string]any{"x": params[0]})
		}, func(int) any { return nil })
		plainProgram, err := expr.Compile(source, expr.Env(map[string]any{}), expr.Optimize(false), plainFn)
		require.NoError(t, err)
		out, err = plainVM.Run(plainProgram, map[string]any{})
		require.NoError(t, err)
		require.Equal(t, []any{11, 30}, out)
	})
}

// TestObserver_Abort 校验观察函数返回错误时运行立即终止且错误原样返回。
func TestObserver_Abort(t *testing.T) {
	sentinel := errors.New("observer sentinel stop")

	var mu sync.Mutex
	calls := map[string]int{}
	env := map[string]any{}
	fooFn := expr.Function("foo", func(params ...any) (any, error) {
		mu.Lock()
		calls["foo"]++
		mu.Unlock()
		return 1, nil
	})
	barFn := expr.Function("bar", func(params ...any) (any, error) {
		mu.Lock()
		calls["bar"]++
		mu.Unlock()
		return 2, nil
	})
	source := `let a = foo(); let b = bar(); a + b`
	program, err := expr.Compile(source, expr.Env(env), expr.Optimize(false), fooFn, barFn)
	require.NoError(t, err)

	// 在 foo 调用完成的事件处终止：bar 绝不能被执行。
	var lastSeq uint64
	observer := func(event vm.Event) error {
		lastSeq = event.Seq
		if event.Opcode == vm.OpCall1 || event.Opcode == vm.OpCallN ||
			event.Opcode == vm.OpCallFast || event.Opcode == vm.OpCall0 {
			loc := event.Location
			if strings.Contains(source[loc.From:loc.To], "foo") {
				return sentinel
			}
		}
		return nil
	}
	out, err := vm.RunObserve(program, env, observer)
	require.Equal(t, sentinel, err, "观察函数的错误必须原样返回，不得包装")
	require.Nil(t, out)

	var fileErr *file.Error
	require.False(t, errors.As(err, &fileErr), "主动终止错误不能是 *file.Error")

	mu.Lock()
	require.Equal(t, 1, calls["foo"])
	require.Equal(t, 0, calls["bar"], "终止后后续调用不得执行")
	mu.Unlock()
	require.Greater(t, lastSeq, uint64(0))

	// 终止发生在嵌套子程序中时，错误必须原样穿透 Go 函数调用边界。
	subProgram, err := expr.Compile(`x + 10`,
		expr.Env(map[string]any{"x": 0}),
		expr.Optimize(false),
	)
	require.NoError(t, err)
	parentVM := &vm.VM{}
	nestedFn := expr.Function("compute", func(params ...any) (any, error) {
		return parentVM.RunNested(subProgram, map[string]any{"x": params[0]})
	}, func(int) any { return nil })
	nestedProgram, err := expr.Compile(`[compute(1), compute(20)]`,
		expr.Env(map[string]any{}),
		expr.Optimize(false),
		nestedFn,
	)
	require.NoError(t, err)

	nestedAbort := func(event vm.Event) error {
		if event.Layer == 1 {
			return sentinel
		}
		return nil
	}
	out, err = parentVM.RunObserve(nestedProgram, map[string]any{}, nestedAbort)
	require.Equal(t, sentinel, err, "嵌套层的终止错误必须原样透传到顶层")
	require.Nil(t, out)
}

// TestObserver_RuntimeError 校验运行时错误仍然转换为绑定源位置的
// *file.Error，且出错指令本身不产生事件；错误类型、顺序与普通执行一致。
func TestObserver_RuntimeError(t *testing.T) {
	program, err := expr.Compile(`a % b`,
		expr.Env(map[string]any{"a": 0, "b": 0}),
		expr.Optimize(false),
	)
	require.NoError(t, err)
	env := map[string]any{"a": 1, "b": 0}

	_, plainErr := vm.Run(program, env)
	require.Error(t, plainErr)
	var plainFileErr *file.Error
	require.True(t, errors.As(plainErr, &plainFileErr))
	require.Equal(t, "runtime error: integer divide by zero", plainFileErr.Message)

	collector := &eventCollector{}
	out, err := vm.RunObserve(program, env, collector.observe)
	require.Error(t, err)
	require.Nil(t, out)
	var observedFileErr *file.Error
	require.True(t, errors.As(err, &observedFileErr))
	require.Equal(t, plainFileErr.Error(), observedFileErr.Error())
	require.Equal(t, plainFileErr.Location, observedFileErr.Location)

	// 出错的 OpModulo 指令不得产生事件。
	moduloIP := -1
	for ip, op := range program.Bytecode {
		if op == vm.OpModulo {
			moduloIP = ip
		}
	}
	require.GreaterOrEqual(t, moduloIP, 0)
	events := collector.snapshot()
	requireContinuousSeq(t, events)
	for _, event := range events {
		require.NotEqual(t, moduloIP, event.IP, "执行失败的指令不得产生事件")
	}

	// Go 函数返回的错误同样必须包装为绑定源位置的 file.Error。
	boom := errors.New("boom")
	failFn := expr.Function("fail", func(params ...any) (any, error) {
		return nil, boom
	})
	failProgram, err := expr.Compile(`fail()`, expr.Env(map[string]any{}), failFn)
	require.NoError(t, err)
	_, err = vm.RunObserve(failProgram, map[string]any{}, func(vm.Event) error { return nil })
	var failFileErr *file.Error
	require.True(t, errors.As(err, &failFileErr))
	require.True(t, errors.Is(err, boom), "file.Error 必须保留对原始错误的包装")
	require.Equal(t, "boom", failFileErr.Message)
}

// TestObserver_OptimizeDifference 校验优化开关产生不同的真实指令事件流，但
// 执行结果一致。
func TestObserver_OptimizeDifference(t *testing.T) {
	optimized, err := expr.Compile(`6 * 7`, expr.Optimize(true))
	require.NoError(t, err)
	unoptimized, err := expr.Compile(`6 * 7`, expr.Optimize(false))
	require.NoError(t, err)

	collectorOpt := &eventCollector{}
	outOpt, err := vm.RunObserve(optimized, nil, collectorOpt.observe)
	require.NoError(t, err)
	collectorNo := &eventCollector{}
	outNo, err := vm.RunObserve(unoptimized, nil, collectorNo.observe)
	require.NoError(t, err)

	require.Equal(t, 42, outOpt)
	require.Equal(t, 42, outNo)

	hasOpcode := func(events []vm.Event, opcode vm.Opcode) bool {
		for _, event := range events {
			if event.Opcode == opcode {
				return true
			}
		}
		return false
	}
	eventsOpt := collectorOpt.snapshot()
	eventsNo := collectorNo.snapshot()
	require.False(t, hasOpcode(eventsOpt, vm.OpMultiply), "常量折叠后不应再执行乘法")
	require.True(t, hasOpcode(eventsNo, vm.OpMultiply), "关闭优化后必须真实执行乘法")
	require.NotEqual(t, len(eventsOpt), len(eventsNo))

	// 两个事件流各自仍然连续，且事件指令确实来自被运行的程序字节码。
	for _, pair := range []struct {
		program *vm.Program
		events  []vm.Event
	}{
		{optimized, eventsOpt},
		{unoptimized, eventsNo},
	} {
		requireContinuousSeq(t, pair.events)
		for _, event := range pair.events {
			require.Equal(t, pair.program.Bytecode[event.IP], event.Opcode)
			require.Equal(t, pair.program.Arguments[event.IP], event.Argument)
		}
	}
}

// TestObserver_Concurrent 校验同一程序并发运行时事件流彼此隔离：每次运行序
// 号独立连续、RunID 唯一、结果一致。
func TestObserver_Concurrent(t *testing.T) {
	program, err := expr.Compile(`map(1..100, # * 2)`)
	require.NoError(t, err)

	const goroutines = 64
	var wg sync.WaitGroup
	var idsMu sync.Mutex
	runIDs := make(map[uint64]struct{}, goroutines)
	errs := make(chan error, goroutines)
	wg.Add(goroutines)
	start := make(chan struct{})

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			collector := &eventCollector{}
			<-start
			out, err := expr.RunObserve(program, nil, collector.observe)
			if err != nil {
				errs <- err
				return
			}
			events := collector.snapshot()
			if len(events) == 0 {
				errs <- fmt.Errorf("事件流为空")
				return
			}
			requireContinuousSeq(t, events)
			id := events[0].RunID
			for _, event := range events {
				if event.RunID != id {
					errs <- fmt.Errorf("同一运行内 RunID 不一致")
					return
				}
			}
			expected := make([]any, 100)
			for j := range expected {
				expected[j] = (j + 1) * 2
			}
			if !reflect.DeepEqual(expected, out) {
				errs <- fmt.Errorf("并发运行结果错误：%v", out)
				return
			}
			idsMu.Lock()
			if _, exists := runIDs[id]; exists {
				idsMu.Unlock()
				errs <- fmt.Errorf("并发运行的 RunID 发生碰撞：%d", id)
				return
			}
			runIDs[id] = struct{}{}
			idsMu.Unlock()
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}
	require.Len(t, runIDs, goroutines)
}

// TestObserver_OldEntryRegression 校验未使用新入口时的兼容性，以及虚拟机复用
// 时观察状态的隔离。
func TestObserver_OldEntryRegression(t *testing.T) {
	// 普通执行与观察执行的返回值、错误类型和错误顺序完全一致。
	cases := []string{
		`1 + 2`,
		`a + b`,
		`map(filter(1..4, # > 1), # * 10)`,
		`a % b`,
		`foo() and false`,
	}
	env := map[string]any{
		"a":   1,
		"b":   0,
		"foo": func() bool { return true },
	}
	for _, code := range cases {
		t.Run(code, func(t *testing.T) {
			program, err := expr.Compile(code,
				expr.Env(map[string]any{"a": 0, "b": 0, "foo": func() bool { return false }}),
				expr.Optimize(false),
			)
			require.NoError(t, err)

			plainOut, plainErr := expr.Run(program, env)
			collector := &eventCollector{}
			observedOut, observedErr := expr.RunObserve(program, env, collector.observe)
			require.Equal(t, plainOut, observedOut)
			require.Equal(t, fmt.Sprintf("%v", plainErr), fmt.Sprintf("%v", observedErr))
			require.Equal(t, reflect.TypeOf(plainErr), reflect.TypeOf(observedErr))
		})
	}

	// expr.Eval 等旧入口行为不受影响。
	out, err := expr.Eval(`1 + 2`, nil)
	require.NoError(t, err)
	require.Equal(t, 3, out)

	// nil 观察函数等价于普通执行。
	program, err := expr.Compile(`1 + 2`)
	require.NoError(t, err)
	out, err = vm.RunObserve(program, nil, nil)
	require.NoError(t, err)
	require.Equal(t, 3, out)

	// nil 程序返回普通错误。
	_, err = vm.RunObserve(nil, nil, func(vm.Event) error { return nil })
	require.Error(t, err)

	// 虚拟机复用：被终止的观察运行之后，普通执行与再次观察都必须是干净状态。
	reuseVM := &vm.VM{}
	sentinel := errors.New("stop")
	abortProgram, err := expr.Compile(`1 + 2`, expr.Optimize(false))
	require.NoError(t, err)
	_, err = reuseVM.RunObserve(abortProgram, nil, func(vm.Event) error {
		return sentinel
	})
	require.Equal(t, sentinel, err)

	out, err = reuseVM.Run(abortProgram, nil)
	require.NoError(t, err)
	require.Equal(t, 3, out)

	collector := &eventCollector{}
	out, err = reuseVM.RunObserve(abortProgram, nil, collector.observe)
	require.NoError(t, err)
	require.Equal(t, 3, out)
	events := collector.snapshot()
	require.Equal(t, uint64(1), events[0].Seq, "下一次运行的序号必须从 1 重新开始")
	requireContinuousSeq(t, events)
}

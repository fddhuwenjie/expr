package vm

import (
	"fmt"
	"sync/atomic"

	"github.com/expr-lang/expr/file"
)

// Event 是一条不可变的指令执行观察事件。
//
// 事件按指令真实执行的顺序产生；条件跳转未命中的分支、短路求值被跳过的
// 操作数以及安全调用中被跳过的链路段都不会产生事件。Event 以值形式交付，
// 其全部字段均为值类型，观察函数对事件的修改不会影响虚拟机内部状态，也不
// 会影响后续事件。
type Event struct {
	// Seq 是本次程序运行内连续的事件序号，从 1 开始逐指令递增。
	// 每次调用 RunObserve 都会重新从 1 开始，互不影响。
	Seq uint64

	// RunID 是本次程序运行的唯一标识。同一程序的并发运行以及对同一虚拟
	// 机的重复运行都会得到不同的 RunID，可用于区分不同运行的事件流。
	RunID uint64

	// Layer 是当前事件所属的调用层标识。顶层程序的事件位于第 0 层；在被
	// 表达式调用的 Go 函数中通过 RunNested 运行另一个已编译程序时，子程
	// 序的事件位于更深一层。
	Layer int

	// IP 是被执行指令在程序字节码中的下标（取值范围与 Program.Bytecode
	// 的下标一致）。
	IP int

	// Opcode 是实际执行的指令。
	Opcode Opcode

	// Argument 是该指令携带的原始整数参数。
	Argument int

	// Location 是该指令对应的表达式源位置。
	Location file.Location

	// StackDepth 是指令执行完成后当前操作数栈的深度。
	StackDepth int
}

// Observer 是事件观察函数。它会在每条指令执行完成后、按真实执行顺序被回调。
//
// 返回非 nil 错误会立即终止本次运行：虚拟机会在当前指令处停止，不再执行后
// 续指令，并且 RunObserve 原样返回该错误（不会包装成 file.Error）。
type Observer func(event Event) error

// observeRunCounter 用于为每次被观察的顶层运行分配全局唯一的 RunID。
var observeRunCounter uint64

// observeSession 承载一次观察运行（含通过 RunNested 进入的嵌套子程序运行）
// 的共享状态。会话只在同步执行的调用链内传递，同一程序的并发运行各自持有
// 独立的会话，因此序号与终止状态天然相互隔离。
type observeSession struct {
	// id 是本次顶层运行的唯一标识。
	id uint64
	// seq 是已发出事件的连续序号，仅在所属运行的执行 goroutine 上变更。
	seq uint64
	// observer 是使用方提供的观察函数；顶层运行与嵌套运行共用同一个函数。
	observer Observer
	// stop 在观察函数返回错误后被设置，用于保证运行立即终止，并让穿过
	// Go 函数调用边界的终止错误原样透传。
	stop error
}

func newObserveSession(observer Observer) *observeSession {
	return &observeSession{
		id:       atomic.AddUint64(&observeRunCounter, 1),
		observer: observer,
	}
}

// emit 按顺序产生一条事件。事件的序号与运行标识在会话内统一分配，保证跨嵌
// 套子程序仍然连续。返回观察函数给出的终止错误（如有）。
func (s *observeSession) emit(event Event) error {
	if s.stop != nil {
		return s.stop
	}
	s.seq++
	event.Seq = s.seq
	event.RunID = s.id
	err := s.observer(event)
	if err != nil {
		s.stop = err
	}
	return err
}

// RunObserve 使用全新的虚拟机运行字节码程序，并通过 observer 按真实执行顺
// 序交付不可变事件。observer 为 nil 时等价于 Run。
func RunObserve(program *Program, env any, observer Observer) (any, error) {
	if program == nil {
		return nil, fmt.Errorf("program is nil")
	}
	vm := VM{}
	return vm.RunObserve(program, env, observer)
}

// RunObserve 在当前虚拟机上运行字节码程序，并通过 observer 按真实执行顺序
// 交付不可变事件。重复使用同一个虚拟机调用本方法时，序号、调用层等观察状
// 态都会在每次运行开始时重置，不会携带上一次运行的任何状态。
//
// observer 返回的错误会被原样作为本方法的错误返回；程序自身的运行时错误仍
// 会按既有行为转换为绑定源位置的 *file.Error。
func (vm *VM) RunObserve(program *Program, env any, observer Observer) (any, error) {
	if observer == nil {
		return vm.Run(program, env)
	}
	if program == nil {
		return nil, fmt.Errorf("program is nil")
	}
	return vm.runSession(program, env, newObserveSession(observer), 0)
}

// RunNested 在当前观察会话内运行另一个已编译程序，子程序事件以更深一层的
// 调用层标识（Layer 加一）与父程序事件共享同一个 RunID 和连续序号。
//
// 本方法用于在被表达式调用的 Go 函数中运行子程序：它会使用独立的虚拟机实
// 例执行，因此不会破坏父虚拟机的栈与变量；事件仍按真实执行顺序插入父程序
// 的事件流中。观察函数在子程序内返回的终止错误同样会立即终止整个运行并原
// 样透传到顶层。
//
// 若当前虚拟机并未处于被观察的运行中，本方法退化为普通的 Run。
func (vm *VM) RunNested(program *Program, env any) (any, error) {
	if program == nil {
		return nil, fmt.Errorf("program is nil")
	}
	child := &VM{}
	if vm.sess == nil {
		return child.Run(program, env)
	}
	if vm.sess.stop != nil {
		// 会话已被观察函数终止：不得再执行任何指令或产生任何事件。
		return nil, vm.sess.stop
	}
	return child.runSession(program, env, vm.sess, vm.obsLayer+1)
}

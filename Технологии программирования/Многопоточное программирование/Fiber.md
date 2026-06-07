==Опр.== **Fiber (Green Thread)** - это легковесные потоки выполнения, работающие полностью в пространстве пользователя (userspace). Каждый fiber обладает собственным полноценным стеком, выделенным в куче или через специальный маппинг памяти.

### Анатомия стека и регистров процессора (повторение)

При вызове обычной функции компилятор генерирует код, управляющий стековым фреймом с помощью ключевых регистров архитектуры x86_64:
- `RSP` (Stack Pointer) - указывает на текущую вершину стека. Стек растёт вниз *(от старших адресов к младшим)*.
- `RBP` (Base/Frame Pointer) - указывает на базу текущего стекового фрейма.
- `RIP` (Instruction Pointer) - содержит адрес инструкции, выполняемой в данный момент.

При вызове функции (инструкция `call`) адрес следующей за ней инструкции автоматически проталкивается на стек в качестве **адреса возврата**.

Регистры процессора разделяются на две фундаментальные группы по соглашению о вызовах:

1. **Caller-saved (Volatile) регистры:** `RAX`, `RCX`, `RDX`, `RSI`, `RDI`, `R8`-`R11`. Вызываемая функция имеет право затирать их значения. Если вызывающему коду важны эти данные, он сам обязан сохранить их на стеке перед вызовом функции.
2. **Callee-saved (Non-volatile) регистры:** `RBX`, `RBP`, `R12`, `R13`, `R14`, `R15`. Если вызываемая функция планирует использовать эти регистры, она обязана сохранить их исходные значения на стеке и восстановить их перед выходом.

### Context Switch

==Опр.== **Переключение контекста fiber (Context Switch)** - это контролируемый переход с одного стека пользователя на другой без вовлечения ядра ОС. С точки зрения компилятора C++, функция переключения контекста выглядит как обычный вызов функции.

> [!important] Важная деталь реализации
> Так как переключение контекста оформлено как стандартный вызов функции `switch_context`, компилятор C++ берет на себя автоматическое сохранение всех _caller-saved_ регистров. Следовательно, в ассемблерном коде функции `switch_context` нам необходимо **вручную сохранить только callee-saved регистры и указатель стека RSP**.

Ниже представлена детальная ассемблерная реализация функции `switch_context` для архитектуры x86_64. Функция принимает два параметра: `from_ctx` (в регистре `RDI`) и `to_ctx` (в регистре `RSI`) в соответствии со стандартным System V AMD64 ABI.
```asm
global switch_context

switch_context:
    ; ------------------------------------------------------------------
    ; ЭТАП 1: Сохранение контекста текущего файбера
    ; ------------------------------------------------------------------
    ; Текущий RIP уже находится на вершине стека благодаря инструкции 'call'
    
    ; Пушим на текущий стек все callee-saved регистры
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    
    ; Сохраняем текущую вершину стека (RSP) в структуру Context первого файбера.
    ; Регистр RDI содержит указатель на Context* from_ctx, то есть &from_ctx->rsp
    mov [rdi], rsp
    
    ; ------------------------------------------------------------------
    ; ЭТАП 2: Активация контекста целевого файбера (to)
    ; ------------------------------------------------------------------
    ; Регистр RSI содержит указатель на Context* to_ctx, то есть &to_ctx->rsp
    ; Загружаем сохраненный RSP целевого файбера в регистр процессора RSP
    mov rsp, [rsi]
    
    ; Теперь мы находимся на стеке нового файбера!
    ; Восстанавливаем callee-saved регистры нового файбера в строго обратном порядке
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    
    ; Instruction 'ret' снимает с вершины нового стека адрес возврата (RIP),
    ; который был сохранен туда ранее, и совершает прыжок по этому адресу.
    ret
```

При создании нового fiber его стек пуст. Чтобы первый вызов `switch_context` на этот fiber не привёл к UB, стек необходимо предварительно разметить («сфабриковать» контекст). Стек fiber выделяется в куче как массив байт фиксированного размера. Нам необходимо разместить элементы на стеке так, чтобы функция `switch_context` смогла выполнить свои инструкции `pop` и `ret`:
```asm
Layout стека файбера после инициализации

[ Старшие адреса (Дно стека) ]
  | ... 
  | [Пользовательские данные / Аргументы лямбды]
  | [Адрес функции-трамплина Trampoline] <-- Сюда укажет 'ret' после pop'ов
  | [Фиктивный R15 (0)]
  | [Фиктивный R14 (0)]
  | [Фиктивный R13 (0)]
  | [Фиктивный R12 (0)]
  | [Фиктивный RBP (0)]
  | [Фиктивный RBX (0)]  <-- Сюда указывает Context::rsp перед первым switch_context
[ Младшие адреса (Вершина стека) ]
```

Реализация инициализации на C++:
```C++
#include <cstdint>
#include <cstdlib>

struct Context {
    void* rsp; // Указатель на вершину стека, где сохранены callee-saved регистры
};

// Прототип ассемблерной функции
extern "C" void switch_context(Context* from, Context* to);

// Функция-трамплин для запуска файбера
void fiber_trampoline() {
    // Получаем указатель на исполняемый файбер из Thread-Local хранилища планировщика
    Fiber* current = Scheduler::Current();
    // Выполняем пользовательскую функцию файбера
    current->RunUserFunction();
    // После завершения функции переводим файбер в состояние dead
    current->SetState(FiberState::DEAD);
    // Добровольно отдаём управление планировщику для финальной утилизации
    Scheduler::TerminateCurrent();
}

void setup_context(Context* ctx, void* stack_bytes, size_t stack_size) {
    // Вычисляем указатель на "дно" стека
    uint8_t* stack_top = static_cast<uint8_t*>(stack_bytes) + stack_size;
    
    // Выравниваем стек по границе 16 байт в соответствии с требованиями ABI
    uint64_t* rsp = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(stack_top) & -16L);
    
    // Кладём адрес возврата для будущей инструкции 'ret' внутри switch_context
    --rsp;
    *rsp = reinterpret_cast<uint64_t>(fiber_trampoline);
    
    // Резервируем место под 6 callee-saved регистров (RBX, RBP, R12-R15)
    for (int i = 0; i < 6; ++i) {
        --rsp;
        *rsp = 0;
    }
    
    // Записываем итоговый указатель вершины стека в объект контекста файбера
    ctx->rsp = static_cast<void*>(rsp);
}
```

### Планировщик (shelduer) для fibers

Планировщик fibers работает полностью в пространстве пользователя на базе пула системных потоков выполнения и полностью контролирует переключение между зелёными потоками и их жизненный цикл.

Fiber на протяжении своего существования может находиться в одном из следующих дискретных состояний:

```C++
enum class FiberState {
    RUNNING,   // Файбер выполняется в данный момент системным потоком
    READY,     // Файбер готов к выполнению и находится в очереди планировщика
    SUSPENDED, // Файбер заблокирован на примитиве синхронизации
    DEAD       // Пользовательская функция завершена, файбер ожидает очистки ресурсов
};
```

![[4_стадии_жизни_файбера.png]]

```C++
class Fiber : public IntrusiveNode {
public:
    Fiber(std::function<void()> func, size_t stack_size = 64 * 1024)
        : user_func_(std::move(func)), stack_size_(stack_size) {
        stack_bytes_ = std::malloc(stack_size_);
        setup_context(&context_, stack_bytes_, stack_size_);
    }

    ~Fiber() {
        std::free(stack_bytes_);
    }

    Context* GetContext() { return &context_; }
    void SetState(FiberState state) { state_ = state; }
    FiberState GetState() const { return state_; }
    void RunUserFunction() { user_func_(); }

private:
    Context context_;
    std::function<void()> user_func_;
    void* stack_bytes_;
    size_t stack_size_;
    FiberState state_ = FiberState::Ready;
};
```

### Реализация функции `yield()` и цикла планировщика

Метод `yield()` заставляет текущий fiber добровольно отказаться от процессора в пользу других fibers, находящихся в состоянии `Ready`.

```C++
class Scheduler {
public:
    static Scheduler* Instance() {
        static thread_local Scheduler scheduler;
        return &scheduler;
    }
    
    static Fiber* Current() { return Instance()->current_fiber_; }
    
    void Yield() {
        Fiber* from = current_fiber_;
        // Переводим текущий файбер в состояние Ready и помещаем в конец очереди задач
        from->SetState(FiberState::Ready);
        run_queue_.PushBack(from);
        
        // Выбираем следующий файбер на исполнение
        Fiber* to = static_cast<Fiber*>(run_queue_.PopFront());
        if (!to) {
            // Если очередь пуста, продолжаем выполнять текущий файбер
            return;
        }
        
        to->SetState(FiberState::Running);
        current_fiber_ = to;
        
        // Совершаем низкоуровневый прыжок контекста
        switch_context(from->GetContext(), to->GetContext());
    }
    
    static void TerminateCurrent() {
        Instance()->ExitCurrent();
    }

private:
    void ExitCurrent() {
        Fiber* from = current_fiber_;
        
        // Извлекаем следующий файбер из очереди
        Fiber* to = static_cast<Fiber*>(run_queue_.PopFront());
        
        if (to) {
            to->SetState(FiberState::Running);
            current_fiber_ = to;
            
            // Важно: файбер 'from' находится в состоянии Terminated
            // Мы переключаем контекст на 'to', но память файбера 'from' ещё жива
            // Планировщик должен удалить стек 'from' позже, находясь на другом стеке
            switch_context(from->GetContext(), to->GetContext());
        } else {
            // Если готовых задач нет, возвращаемся в контекст основного потока ОС
            current_fiber_ = nullptr;
            switch_context(from->GetContext(), &main_context_);
        }
    }
    
    IntrusiveQueue run_queue_;
    Fiber* current_fiber_ = nullptr;
    Context main_context_;
};
```

> [!warning] Проблема утилизации стека завершенного файбера
> 
> Нельзя вызвать `delete` или `free()` для объекта `Fiber` внутри кода самого этого файбера. Если освободить память стека, на котором процессор выполняет текущую инструкцию, произойдет мгновенный крах системы (Segmentation Fault). Планировщик решает эту проблему методом отложенного удаления (Garbage Collection): уничтожение завершённых объектов файберов происходит в основном цикле планировщика на стеке системного потока.

---

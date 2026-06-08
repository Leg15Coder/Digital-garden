==Опр.== **Lock-freedom** - свойство алгоритма, при котором система в целом гарантирует прогресс. Это означает, что если несколько потоков одновременно пытаются выполнить операцию над структурой данных, то хотя бы один из них гарантированно завершит свою работу за конечное число шагов.

- **Особенность**: Отдельные потоки могут голодать (т.е. бесконечно крутиться в цикле из-за того, что другие потоки их опережают и меняют состояние структуры), однако система не зависает.

==Опр.== **Lock-freedom** - свойство алгоритма, при котором каждый отдельный поток гарантированно завершает свою операцию за конечное число собственных шагов, независимо от поведения, скорости и засыпания других потоков.

- **Особенность**: Здесь полностью отсутствуют потенциально бесконечные циклы конкуренции. Потоки никогда не голодают. Любой wait-free алгоритм является lock-free, но не наоборот.

## Lock-free стек (Стек Трайбера)

Стек Трайбера представляет собой односвязный список, в котором вершина `head` является атомарным указателем. Все операции происходят исключительно с головой списка.

```C++
#include <atomic>
#include <optional>

template <typename T>
class TreiberStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(const T& val) : data(val), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};

public:
    ~TreiberStack() {
        Node* curr = head.load(std::memory_order_relaxed);
        while (curr) {
            Node* next_node = curr->next;
            delete curr;
            curr = next_node;
        }
    }

    void push(const T& val) {
        Node* new_node = new Node(val);  // ОШИБКА: блокирующий оператор
        // Загружаем текущую голову
        new_node->next = head.load(std::memory_order_relaxed);
        
        // Пытаемся атомарно подменить голову. 
        // Если head изменился, CAS обновит new_node->next текущим значением head
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }

    std::optional<T> pop() {
        Node* curr = head.load(std::memory_order_relaxed);
        
        // Пытаемся сместить голову на следующий элемент
        while (curr && !head.compare_exchange_weak(curr, curr->next));
        
        if (!curr) {
            return std::nullopt; // Стек пуст
        }

        T res = std::move(curr->data);
        
        // ОШИБКА: Прямой вызов delete здесь вызовет Use-After-Free!
        delete curr;
        return res;
    }
};
```

Чтобы избежать ошибок с Use-After-Free и избавления от блокирующего оператора new в push в неблокирующих структурах часто используют пул переиспользования памяти. Поскольку память узлов физически никогда не возвращается ОС через `delete`, чтение `curr->next` всегда безопасно с точки зрения ОС:
```c++
#include <atomic>
#include <optional>
#include <utility>

template <typename T>
class TreiberStackWithFreeList {
private:
    struct Node {
        T data;
        Node* next;
        Node() : next(nullptr) {}
        Node(const T& val) : data(val), next(nullptr) {}
    };

    std::atomic<Node*> head{nullptr};
    
    // Пул для переиспользования освобожденных нод (тоже lock-free стек)
    std::atomic<Node*> free_list{nullptr};

    Node* allocate_node(const T& val) {
        Node* old_free = free_list.load(std::memory_order_relaxed);
        while (old_free && !free_list.compare_exchange_weak(old_free, old_free->next));
        if (old_free) {
            old_free->data = val;
            old_free->next = nullptr;
            return old_free;
        }
        return new Node(val);
    }

    void deallocate_node(Node* node) {
        node->next = free_list.load(std::memory_order_relaxed);
        while (!free_list.compare_exchange_weak(node->next, node));
    }

public:
    ~TreiberStackWithFreeList() {
        auto clear_list = [](std::atomic<Node*>& list_head) {
            Node* curr = list_head.load(std::memory_order_relaxed);
            while (curr) {
                Node* next_node = curr->next;
                delete curr;
                curr = next_node;
            }
        };
        clear_list(head);
        clear_list(free_list);
    }

    void push(const T& val) {
        Node* new_node = allocate_node(val);
        new_node->next = head.load(std::memory_order_relaxed);
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }

    std::optional<T> pop() {
        Node* curr = head.load(std::memory_order_acquire);
        
        while (curr) {
            Node* next_node = curr->next; 
            
            if (head.compare_exchange_weak(curr, next_node)) {
                T res = std::move(curr->data);
                deallocate_node(curr); // Нода уходит в пул свободных нод
                return res;
            }
        }
        return std::nullopt;
    }
};
```

## Проблема ABA

Проблема возникает в CAS-ориентированных алгоритмах при повторном использовании адресов памяти:
1. Поток 1 пытается сделать `pop()`. Он считывает текущую голову списка `head` (значение **A**) и указатель на следующий элемент `curr->next` (значение **B**). Поток 1 планирует перевести `head` из состояния **A** в состояние **B**.
2. Поток 1 вытесняется планировщиком ОС непосредственно перед вызовом `compare_exchange`.
3. Поток 2 просыпается и делает `pop()`, успешно удаляя узел **A**.
4. Поток 2 делает ещё один `pop()`, удаляя узел **B** и освобождая память.
5. Поток 2 делает `push()` нового элемента. Аллокатор памяти переиспользует адрес только что освобожденного узла **A**. Теперь узел **A** снова находится на вершине стека, но его внутреннее поле `next` теперь указывает на совершенно другой узел (или мусор), а не на **B**.
6. Поток 1 просыпается и выполняет `compare_exchange_weak(A, B)`. Инструкция видит, что текущее значение `head` равно **A**, считает, что ничего не изменилось, и успешно прописывает в `head` невалидный указатель **B**. Структура данных разрушена.

### Решение проблемы: Tagged Pointers

Для устранения ABA-проблемы к указателю привязывается счётчик модификаций (версия/тег). Каждый раз при изменении указателя тег инкрементируется. Таким образом, состояние меняется по схеме $A_1 \to B_2 \to A_3$. Когда Поток 1 попытается сделать CAS с ожидаемым значением $A_1$, проверка провалится, так как текущее значение - $A_3$. Для реализации требуется **Double-Word CAS (DWCAS)** *(инструкция `cmpxchg16b` на архитектуре x86-64)*, позволяющая атомарно изменять 16 байт памяти за один такт.

```C++
#include <atomic>
#include <cstdint>

template <typename T>
class TaggedPointer {
public:
    struct Node;

    struct TaggedRef {
        Node* ptr;
        uintptr_t tag;
    };

    struct Node {
        T data;
        TaggedRef next;
        Node(const T& val) : data(val), next{nullptr, 0} {}
    };
};
```

## Lock-free очередь (Очередь Майкла-Скотта)

Очередь Майкла-Скотта (Michael-Scott Queue) использует односвязный список с двумя указателями: `head` (откуда забирают) и `tail` (куда добавляют).

#### Ключевые идеи реализации:

1. **Фиктивный узел**: Очередь всегда инициализируется одним пустым узлом. `head` всегда указывает на этот фиктивный узел, а реальный первый элемент находится в `head->next`. Это исключает состояние, когда очередь абсолютно пуста, убирая необходимость сложной взаимной блокировки указателей `head` и `tail`.

2. **Helping**: Из-за того, что невозможно одной аппаратной инструкцией обновить и `tail->next`, и сам `tail`, операция вставки разделена на два шага. Если поток видит, что `tail->next != nullptr`, это значит, что другой поток застрял между шагами. Текущий поток _помогает_ продвинуть `tail` вперед, обеспечивая свойство lock-free.

### Реализация на C++:
```C++
#include <atomic>
#include <optional>

template <typename T>
class MichaelScottQueue {
private:
    struct Node;

    struct TaggedPtr {
        Node* ptr;
        uintptr_t tag;
    };

    struct Node {
        T data;
        std::atomic<TaggedPtr> next;
        Node() : next(TaggedPtr{nullptr, 0}) {}
        Node(const T& val) : data(val), next(TaggedPtr{nullptr, 0}) {}
    };

    std::atomic<TaggedPtr> head;
    std::atomic<TaggedPtr> tail;

public:
    MichaelScottQueue() {
        Node* dummy = new Node();
        TaggedPtr init_ptr = {dummy, 0};
        head.store(init_ptr, std::memory_order_relaxed);
        tail.store(init_ptr, std::memory_order_relaxed);
    }

    void enqueue(const T& val) {
        Node* new_node = new Node(val);
        while (true) {
            TaggedPtr curr_tail = tail.load(std::memory_order_acquire);
            TaggedPtr next_ptr = curr_tail.ptr->next.load(std::memory_order_acquire);
            TaggedPtr check_tail = tail.load(std::memory_order_acquire);

            if (curr_tail.ptr == check_tail.ptr && curr_tail.tag == check_tail.tag) {
                if (next_ptr.ptr == nullptr) {
                    // Шаг 1: Пытаемся привязать узел к концу списка
                    TaggedPtr new_next = {new_node, next_ptr.tag + 1};
                    if (curr_tail.ptr->next.compare_exchange_strong(next_ptr, new_next)) {
                        // Шаг 2: Пытаемся передвинуть tail на новый узел
                        TaggedPtr new_tail = {new_node, curr_tail.tag + 1};
                        tail.compare_exchange_strong(curr_tail, new_tail);
                        return;
                    }
                } else {
                    // Helping: Очередь в промежуточном состоянии, помогаем продвинуть tail
                    TaggedPtr new_tail = {next_ptr.ptr, curr_tail.tag + 1};
                    tail.compare_exchange_strong(curr_tail, new_tail);
                }
            }
        }
    }

    std::optional<T> dequeue() {
        while (true) {
            TaggedPtr curr_head = head.load(std::memory_order_acquire);
            TaggedPtr curr_tail = tail.load(std::memory_order_acquire);
            TaggedPtr next_ptr = curr_head.ptr->next.load(std::memory_order_acquire);
            TaggedPtr check_head = head.load(std::memory_order_acquire);

            if (curr_head.ptr == check_head.ptr && curr_head.tag == check_head.tag) {
                if (curr_head.ptr == curr_tail.ptr) {
                    if (next_ptr.ptr == nullptr) {
                        return std::nullopt; // Очередь пуста
                    }
                    // Helping: tail отстал от head, продвигаем его
                    TaggedPtr new_tail = {next_ptr.ptr, curr_tail.tag + 1};
                    tail.compare_exchange_strong(curr_tail, new_tail);
                } else {
                    // Читаем данные из следующего за head узла (он станет новым фиктивным)
                    T val = next_ptr.ptr->data;
                    TaggedPtr new_head = {next_ptr.ptr, curr_head.tag + 1};
                    
                    if (head.compare_exchange_strong(curr_head, new_head)) {
                        // Память curr_head.ptr логически удалена. 
                        // Настоящее освобождение требует безопасного менеджера памяти.
                        return val;
                    }
                }
            }
        }
    }
};
```

## Проблема управления памятью в lock-free структурах

В блокирующих структурах узел удаляется под мьютексом: мы уверены, что никто больше не держит ссылку на него. В lock-free алгоритмах поток может считать указатель на узел, затем быть вытесненным. Другой поток удалит этот узел и вызовет `delete`. Когда первый поток проснётся и попытается прочитать `curr->next`, произойдет обращение к освобождённой памяти (`Segmentation Fault`).

### Методы решения проблемы

1. **Free List**:
    Вместо физического возврата памяти операционной системе через `delete`, удалённые узлы переносятся в lock-free стек свободных узлов. При выделении новой памяти структура сначала пытается забрать узел из этого пула. Это частично нивелирует ABA-проблему и защищает от падения ОС по защите памяти, но может приводить к неограниченному росту пула.
    
2. **Hazard Pointers**:
    Каждый читающий поток перед началом работы с узлом записывает его адрес в глобальный, видимый для всех массив «опасных» указателей (`Hazard Pointers`). Это явный сигнал: _«Я сейчас читаю этот узел, не удаляйте его»_. Поток, удаляющий ноду, убирает её из структуры данных, но помещает в свой приватный список отложенного удаления (`Retired List`). Периодически поток сканирует свой список и физически очищает только те узлы, которых нет ни в одном Hazard Pointer.
    
3. **Epoch-Based Reclamation**:
    Время работы делится на глобальные эпохи. Каждый поток при входе в критический участок заявляет о своей активности в текущей эпохе. При удалении элемента он помещается в корзину текущей эпохи. Физическое освобождение корзины эпохи $N$ происходит только тогда, когда все активные потоки гарантированно перешли в эпоху $N+1$. Метод имеет крайне низкие накладные расходы, но если один поток уснёт, очистка памяти заблокируется для всей системы.
    
## Класс `std::atomic_shared_ptr`
### В чём отличие от обычного `std::shared_ptr`?

- В обычном `std::shared_ptr` внутренние счётчики ссылок изменяются атомарно, что делает безопасным создание копий умного указателя в разных потоках. Однако сам управляющий объект *(пара из указателя на данные и указателя на контрольный блок)* не является атомарным. Параллельное чтение и запись в одну и ту же переменную `std::shared_ptr` из разных потоков - это Data Race и UB.
- `std::atomic_shared_ptr` *(введённый как `std::atomic<std::shared_ptr<T>>` в C++20\)* делает атомарными операции над самим объектом-указателем (`load`, `store`, `compare_exchange`).

### В чём идея реализации?

Главная сложность реализации - атомарно считать указатель на контрольный блок и успеть увеличить в нем счётчик ссылок до того, как другой поток вызовет `store(nullptr)` и уничтожит этот контрольный блок. Реализация требует синергии двух подходов:
1. **Double-Word CAS (128-битный CAS)**: Используется для атомарного обновления структуры из двух указателей *(указатель на данные + указатель на контрольный блок)* за одну инструкцию.
2. **Split Reference Counting**:
    Счётчик ссылок разделяется на две части. Часть счётчика упаковывается непосредственно внутрь самого 16-байтового атомарного указателя *(в составе свободного пространства тега)*. При чтении поток атомарно увеличивает внешний счётчик прямо в указателе с помощью DWCAS, тем самым безопасно резервируя за собой право доступа. И только после этого поток переносит накопленный инкремент внутрь основного контрольного блока в куче.

### Можно ли сделать `atomic_shared_ptr` полностью lock-free?

**Да, можно**, но это напрямую зависит от аппаратной платформы и компилятора:
- На архитектурах, поддерживающих 128-битные атомарные инструкции, и при использовании компилятором техники _Split Reference Counting_, данный класс работает с честной **Lock-free гарантией**. В кодовой базе это проверяется через макрос свойства `std::atomic<std::shared_ptr<T>>::is_always_lock_free`.
- На архитектурах, где нет поддержки 128-битного CAS (или при работе с `weak_ptr`), библиотека компилятора под капотом откатывается к использованию скрытого внутреннего пула мьютексов (`striped locking`) для защиты указателей. В этом сценарии структура **не является lock-free**, несмотря на атомарный интерфейс.

---

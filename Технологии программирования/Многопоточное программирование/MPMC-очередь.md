==Опр.== **MPMC-очередь** - это высокопроизводительный [[Кольцевой буфер|циклический буфер]], с которым параллельно работают несколько продюсеров (потоков записи) и несколько консьюмеров (потоков чтения) без использования единого глобального мьютекса.

## 1. Реализация

Основная идея заключается в том, что **каждая ячейка массива является независимым примитивом синхронизации**.

### Устройство СД:

1. Массив ячеек `Cell` фиксированного размера $N$ (где $N$ - степень двойки для быстрой битовой маски `pos & (N - 1)`).

2. Каждая ячейка содержит:
    - Полезную нагрузку (`T data`).
    - Атомарный счётчик последовательности: `std::atomic<size_t> sequence`.

3. Глобальные указатели (билеты):
    - `std::atomic<size_t> enqueue_pos_{0}` (куда пытается писать следующий продюсер).
    - `std::atomic<size_t> dequeue_pos_{0}` (откуда пытается читать следующий консьюмер).

**Начальное состояние:** Для каждой ячейки $i$ от $0$ до $N-1$ значение `sequence` инициализируется её индексом: `cell[i].sequence = i`.

```c++
template <typename T>
class MPMCQueue {
private:
    // Каждая ячейка буфера выравнивается по размеру кэш-линии,
    // чтобы предотвратить деструктивное ложное разделение кэша
    struct alignas(std::hardware_destructive_interference_size) Cell {
        std::atomic<size_t> sequence;
        T data;
    };

    std::vector<Cell> buffer_;
    size_t const buffer_mask_;

    // Глобальные указатели (билеты) тоже выравниваем, чтобы они не пересекались 
    // в кэше процессора друг с другом и с телом буфера.
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> enqueue_pos_{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> dequeue_pos_{0};

public:
    // Размер очереди (capacity) строго обязан быть степенью двойки!
    explicit VyukovMPMCQueue(size_t capacity)
        : buffer_(capacity),
          buffer_mask_(capacity - 1) {
        for (size_t i = 0; i < capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool Push(T const& data) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        while (true) {
            cell = &buffer_[pos & buffer_mask_];
            // Считываем sequence с барьером acquire. Это гарантирует, что если мы увидим
            // готовность ячейки, то последующая запись в неё не обгонит это чтение.
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                // Пытаемся забрать «билет» (индекс). Указатели билетов изменяются через relaxed,
                // так как они нужны только для распределения работы, а не для синхронизации данных.
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // Очередь полна (продюсер ушёл на круг вперёд, а консьюмер ещё не вычитал данные)
                return false;
            } else {
                // Поток отстал, перечитываем актуальное значение билета
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        // Записываем полезную нагрузку
        cell->data = data;
        
        // Сигнализируем консьюмеру о готовности данных с помощью release-барьера.
        // Запись данных гарантированно завершится до того, как sequence обновится.
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Перегрузка Enqueue для поддержки rvalue-ссылок (move-семантика)
    bool Enqueue(T&& data) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        while (true) {
            cell = &buffer_[pos & buffer_mask_];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }

        cell->data = std::move(data);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Метод Dequeue: Извлечение элемента консьюмером
    bool Dequeue(T& data) {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        
        while (true) {
            cell = &buffer_[pos & buffer_mask_];
            // Считываем статус ячейки. Acquire гарантирует, что чтение cell->data 
            // произойдёт строго после того, как мы убедимся в валидности sequence.
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                // Пытаемся продвинуть dequeue_pos_ вперёд
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                // Очередь пуста (продюсер ещё не записал данные в этот слот)
                return false;
            } else {
                // Перечитываем актуальный билет консьюмера
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }

        // Забираем данные
        data = std::move(cell->data);
        
        // Переводим ячейку в состояние «свободно для следующего круга».
        // Использован release, чтобы чтение данных не переставилось процессором после этой строчки.
        // pos + buffer_mask_ + 1 эквивалентно pos + capacity, то есть сдвигает sequence на полный оборот вперёд.
        cell->sequence.store(pos + buffer_mask_ + 1, std::memory_order_release);
        return true;
    }
};
```

---














## 2. Пошаговые сценарии работы алгоритма

### Сценарий 1: Добавление элемента продюсером (`Enqueue`)

Продюсер хочет записать данные. Ему нужно занять слот и гарантировать, что этот слот пуст и готов именно для его «круга» (оборота циклического буфера).

1. Поток считывает глобальный индекс `pos = enqueue_pos_.load(relaxed)`.
    
2. Он находит нужную ячейку в кольцевом буфере: `cell = &buffer_[pos & buffer_mask_]`.
    
3. Поток считывает `seq = cell->sequence.load(acquire)`.
    
4. Вычисляется разность: `diff = seq - pos`.
    
5. **Анализ состояний (`diff`):**
    
    - **`diff == 0` (Ячейка готова):** Это значит, что ячейка пуста и мы пришли вовремя. Поток пытается зарезервировать этот билет за собой с помощью `compare_exchange_weak(pos, pos + 1)`.
        
        - _Если CAS успешен:_ Поток выходит из цикла, записывает данные в `cell->data`, а затем обновляет статус ячейки: `cell->sequence.store(pos + 1, release)`. Этот инкремент на 1 служит явным сигналом для консьюмера: «Данные готовы».
            
        - _Если CAS провалился:_ Другой продюсер успел забрать этот билет раньше. Наш поток обновляет `pos` и уходит на следующую итерацию цикла.
            
    - **`diff < 0` (Очередь полна):** Это означает, что `pos` ушел вперед (буфер завершил оборот), а ячейка все еще содержит старые данные, которые консьюмеры не успели вычитать. Метод возвращает `false` (или уходит в spin-wait/blocking).
        
    - **`diff > 0`:** Поток отстал и прочитал старое состояние, цикл продолжается.
        

### Сценарий 2: Извлечение элемента консьюмером (`Dequeue`)

Консьюмер хочет забрать данные. Он должен убедиться, что продюсер уже завершил запись в этот слот.

1. Поток считывает глобальный индекс `pos = dequeue_pos_.load(relaxed)`.
    
2. Находит ячейку: `cell = &buffer_[pos & buffer_mask_]`.
    
3. Считывает `seq = cell->sequence.load(acquire)`.
    
4. Вычисляется разность: `diff = seq - (pos + 1)`.
    
5. **Анализ состояний (`diff`):**
    
    - **`diff == 0` (Данные готовы для чтения):** Напомним, продюсер после записи выставил `sequence` в `pos_enqueue + 1`. Если это значение совпадает с нашим `pos + 1`, значит, запись завершена. Поток пытается продвинуть `dequeue_pos_` через `compare_exchange_weak(pos, pos + 1)`.
        
        - _Если CAS успешен:_ Поток забирает данные из `cell->data` и переводит ячейку в состояние «свободно для следующего круга продюсеров», записывая: `cell->sequence.store(pos + buffer_mask_ + 1, release)`.
            
    - **`diff < 0` (Очередь пуста):** Продюсер еще не успел дойти до этой ячейки или не закончил запись (`sequence` еще не инкрементирован). Метод возвращает `false`.
        

## 3. Критические аппаратные нюансы и ошибки для экзамена

### Аппаратная оптимизация: Борьба с False Sharing (Ложным разделением кэша)

Это один из самых частых вопросов на зачете по 6-й неделе.

- **Суть проблемы:** Структура `Cell` содержит только атомик `sequence` и данные `data`. Если размер данных небольшой (например, указатели), то несколько ячеек `Cell` гарантированно попадут на одну и ту же **кэш-линию процессора** (размер которой обычно 64 байта).
    
- Когда Продюсер 1 пишет в `cell[0].sequence`, а Продюсер 2 параллельно пишет в `cell[1].sequence`, их ядра процессора начинают непрерывно инвалидировать кэши друг друга (эффект _Cache Line Bouncing_ согласно протоколу MESI). Производительность падает в разы, межпроцессорная шина блокируется.
    
- **Исправление:** Необходимо принудительно выравнивать каждую ячейку `Cell` по размеру кэш-линии. В C++17 для этого используется макрос выравнивания:
    
    C++
    
    ```
    struct alignas(std::hardware_destructive_interference_size) Cell {
        std::atomic<size_t> sequence;
        T data;
    };
    ```
    

### Важность барьеров памяти (Memory Ordering)

Почему нельзя сделать все операции `relaxed`?

- Продюсер пишет в `cell->data`, а затем делает `cell->sequence.store(..., release)`. Барьер `release` гарантирует, что процессор и компилятор ни в коем случае не переставят запись данных _после_ записи в `sequence`.
    
- Консьюмер делает `cell->sequence.load(..., acquire)`. Барьер `acquire` гарантирует, что чтение `cell->data` произойдет строго _после_ того, как мы убедились в готовности `sequence`.
    
- Если нарушить эти барьеры, консьюмер может прочитать «мусор» или старые данные из `cell->data`, даже если `sequence` уже будет равен корректному значению.
    

## Экзаменационный чек-лист (Ловушки профессора)

1. **Почему в алгоритме Вьюкова глобальные `enqueue_pos_` и `dequeue_pos_` изменяются с помощью `memory_order_relaxed`, а не `seq_cst`?**
    
    - _Ответ:_ Потому что эти указатели служат лишь для распределения «билетов» (индексов) между потоками. Сама синхронизация полезной нагрузки (данных) жестко привязана к атомикам `sequence` внутри конкретных ячеек с помощью пар `acquire-release`. Глобальный порядок здесь не требуется.
        
2. **Что произойдет, если в Condition Variable инкрементировать счетчик поколений (`generation_`) вне мьютекса в методе `NotifyOne`?**
    
    - _Ответ:_ Это приведет к Data Race (гонке по данным), если в этот же момент ждущий поток читает `generation_` внутри `Wait` перед тем, как отпустить мьютекс. Любая модификация разделяемой переменной, которая может пересечься с несинхронизированным чтением, ломает инварианты модели памяти.
        
3. **Зачем в формуле Вьюкова для Dequeue вычитается именно `(pos + 1)`, а не просто `pos`?**
    
    - _Ответ:_ Продюсер, завершив запись в ячейку с абсолютным индексом `pos`, записывает в её `sequence` значение `pos + 1`. Следовательно, консьюмер, приходя на этот же абсолютный индекс `pos`, ожидает увидеть в ячейке именно `pos + 1`, что подтверждает успешное окончание записи.

---

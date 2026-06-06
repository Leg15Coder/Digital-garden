Чтобы избавиться от [[Busy Loop Polling Server|Busy Loop]], нам нужен механизм, позволяющий сказать ядру ОС: _"Усыпи мой поток и разбуди, только когда хотя бы на одном из этих N сокетов появятся данные"_. Для этого служат мультиплексоры I/O.

## 1. Системный вызов `select`

Исторически первый инструмент. Ядру передаются битовые маски сокетов (`fd_set`) для отслеживания.

**Проблемы:** 
* Максимальный файловый дескриптор жёстко ограничен константой `FD_SETSIZE` (обычно 1024). Для решения проблемы C10k не подходит в принципе.
- На каждый вызов приходится заново копировать битовые маски из user-space в ядро и обратно, так как ядро их затирает.
- После возврата `select` потоку нужно линейно перебрать все 1024 бита ($O(N)$), чтобы понять, какой именно сокет готов.

## 2. Системный вызов `poll`

Исправляет ограничение `select`. Принимает массив структур `pollfd`.
- **Плюсы:** Нет ограничения в 1024 сокета.
- **Проблемы:** Огромные накладные расходы на копирование. Если клиентов 10 000, каждый вызов `poll` требует скопировать весь этот массив в ядро. Найти сработавший сокет всё ещё требует итерации по всему массиву ($O(N)$).

## 3. Системный вызов `epoll`

Современный стандарт для высокопроизводительных серверов на Linux. В отличие от предыдущих вызовов, `epoll` сохраняет состояние внутри ядра.

- `epoll_create()`: Создаёт внутри ядра структуру (на базе красно-чёрного дерева для быстрого поиска и очереди готовых событий).
- `epoll_ctl()`: Добавляет, изменяет или удаляет сокеты для мониторинга ($O(\log N)$). Вызывается только один раз при подключении/отключении клиента. Не нужно копировать весь список каждый раз!
- `epoll_wait()`: Блокирует поток до появления активности. **Самое важное:** возвращает массив _только тех сокетов, на которых реально произошли события_ ($O(K)$, где $K$ - число активных сокетов). Никакого $O(N)$ поиска!

#### Реализация многопоточного сервера через `epoll` (Паттерн Reactor):

Один выделенный поток (Reactor) крутит цикл с `epoll_wait()`. Как только `epoll_wait` возвращает список "готовых" дескрипторов, Reactor не обрабатывает их сам, а формирует задачи (например, `ReadTask(fd)`) и кладет их в очередь Thread Pool. Рабочие потоки из пула достают задачу, делают один успешный (неблокирующий) `read`, так как epoll уже гарантировал наличие данных, и отправляют ответ.

```cpp
constexpr int MAX_EVENTS = 64;

int main() {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    // ... (bind, listen) ...

    int epfd = epoll_create1(0); // Создаём epoll

    epoll_event ev{};
    ev.events = EPOLLIN; // Нас интересует чтение (входящие соединения)
    ev.data.fd = server_sock;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server_sock, &ev); // Добавляем сервер в epoll

    std::vector<epoll_event> events(MAX_EVENTS);

    while (true) {
        // Блокируемся до появления событий. Возвращает количество готовых FD
        int ready = epoll_wait(epfd, events.data(), MAX_EVENTS, -1);
        if (ready < 0) break;

        // Итерируемся ТОЛЬКО по готовым событиям (O(K))
        for (int i = 0; i < ready; ++i) {
            int fd = events[i].data.fd;

            if (fd == server_sock) {
                // Новое подключение
                int client_fd = accept(server_sock, nullptr, nullptr);
                
                epoll_event client_ev{};
                client_ev.events = EPOLLIN; // В многопоточном пуле добавили бы | EPOLLET
                client_ev.data.fd = client_fd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &client_ev);
            } else {
                // Готово для чтения
                char buffer[1024];
                ssize_t bytes = read(fd, buffer, sizeof(buffer));
                if (bytes > 0) {
                    // Эхо
                    write(fd, buffer, bytes);
                } else {
                    // Клиент отключился. Закрытие FD автоматически удаляет его из epoll
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                    close(fd);
                }
            }
        }
    }
    close(server_sock);
    close(epfd);
}
```

## 4. Механизм `io_uring`

**Проблема `epoll`:** Это **синхронный** механизм. `epoll` только говорит: _"Данные есть, иди читай"_. Потоку всё равно приходится делать syscall `read`, переключать контекст в режим ядра, чтобы ядро скопировало данные из буфера сетевой карты в буфер приложения.

`io_uring` - это **истинно асинхронный** ввод-вывод. Здесь мы говорим ядру: _"Вот буфер в user-space. Самостоятельно считай данные из этого сокета сюда и сообщи мне, когда закончишь"_.

**Фундаментальное отличие `io_uring` от `epoll`:**
При инициализации `io_uring` ядро через `mmap` отображает участок оперативной памяти напрямую в адресное пространство процесса *(Shared Memory)*. В этой памяти создаётся два кольцевых буфера:
1. **SQ (Submission Queue - Очередь отправки):** Сюда приложение кладет задания для ядра (SQE - Submission Queue Entries), например "сделай read на этом сокете в этот буфер".
2. **CQ (Completion Queue - Очередь завершения):** Сюда ядро кладет результаты (CQE), когда I/O операция физически завершена.

Формат SQE:
```c
struct request_info {
    int type; // ACCEPT, READ, WRITE...
    int fd;
    char buffer[];
    int buffer_size;
};
```

- Синхронизация указателей головы и хвоста буферов-очередей происходит через барьеры памяти atomic (Acquire/Release семантика) без системных блокировок.

### +ы & -ы `io_uring`


> [!success] Преимущества
> 1. **Отсутствие системных вызовов**. Благодаря shared memory можно отправлять сотни запросов и забирать сотни результатов _вообще без переключений контекста_, что даёт колоссальную пропускную способность.
> 2. **Унифицированность**. `io_uring` идеально и единообразно асинхронно работает как с сетью, так и с диском.
> 3. **Работа с батчами**. Можно положить батч запросов `read` и `write` в SQ и сделать всего один вызов `io_uring_enter`.

> [!failure] Недостатки
> 4. Требует *(относительно)* новых версий ядра ОС.
> 5. Сложнее в настройке и использовании (много ручной работы с барьерами памяти).
> 6. Имеет проблемы с безопасностью *(в контейнерах Kubernetes/Docker доступ к io_uring часто по умолчанию заблокирован из-за прямого доступа к памяти ядра)*.

### Реализация сервера с `io_uring`

```c++
void run_server(int server_fd) {
    struct io_uring ring;
    io_uring_queue_init(256, &ring, 0); // Инициализация очередей

    // 1. Просим ядро сделать асинхронный accept
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    struct request_info *req = malloc(sizeof(*req));
    req->type = 1; // ACCEPT
    req->fd = server_fd;

    io_uring_prep_accept(sqe, server_fd, NULL, NULL, 0);
    io_uring_sqe_set_data(sqe, req); // Привязываем наш контекст
    io_uring_submit(&ring);          // Уведомляем ядро

    struct io_uring_cqe *cqe;
    while (1) {
        // Ждем результата от ядра
        io_uring_wait_cqe(&ring, &cqe);
        struct request_info *completed_req = io_uring_cqe_get_data(cqe);
        int res = cqe->res; // Для accept() это client_fd

        if (completed_req->type == 1 && res > 0) { // Это был accept
            int client_fd = res;
            
            // 2. Сразу формируем новую задачу READ для этого клиента
            struct io_uring_sqe *read_sqe = io_uring_get_sqe(&ring);
            struct request_info *read_req = malloc(sizeof(*read_req));
            read_req->type = 2; // READ
            read_req->fd = client_fd;

            // Говорим ядру: прочитай данные ПРЯМО В read_req->buffer!
            io_uring_prep_read(read_sqe, client_fd, read_req->buffer, 1024, 0);
            io_uring_sqe_set_data(read_sqe, read_req);
            
            // И перезаряжаем ACCEPT для новых клиентов...
            // ...
            io_uring_submit(&ring);
        } else if (completed_req->type == 2 && res > 0) {
            // READ завершился, данные УЖЕ в completed_req->buffer
            // Формируем задачу WRITE...
        }

		// Сдвигаем Head в CQ (отмечаем как обработанное)
        io_uring_cqe_seen(&ring, cqe);
        free(completed_req);
    }
}
```

---

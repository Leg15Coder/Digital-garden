### Рассмотрим большинство возможных [[Операторы в SQL|операторов]] в [[SQL]]

#### 1. [[Виды операторов в SQL#^f4a3bd|DDL]] — Операторы определения данных

##### `CREATE`

Создаёт новую таблицу, базу данных, представление, индекс и т. д.

- Можно дописать `IF EXISTS/IF NOT EXISTS` чтобы выполнять только если создаваемый объект уже есть/создаваемого объекта ещё нет

```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    position VARCHAR(50),
    salary DECIMAL(10, 2)
);

CREATE VIEW employees_count_view;
CREATE INDEX idx_employees_name ON employees(name);
CREATE FUNCTION get_bonus(salary DECIMAL) RETURNS DECIMAL;
```


##### `ALTER`

Изменяет существующую структуру таблицы.

```sql
ALTER TABLE employees ADD COLUMN hire_date DATE;
ALTER TABLE employees DROP COLUMN position;
ALTER TABLE employees RENAME TO workers;
```

  
##### `DROP`

Удаляет объекты базы данных, такие как таблицы, базы данных, представления.

```sql
DROP TABLE employees;
DROP DATABASE warehouse_db;
DROP INDEX idx_employees_name;
```


##### `TRUNCATE`

Очищает таблицу, удаляя все строки без логирования (необратимо, быстрее `DELETE`).

```sql
TRUNCATE TABLE employees;
```

  
#### 2. [[Виды операторов в SQL#^5746a5|DML]] — Операторы управления данными

##### `SELECT`

^b14899

Извлекает данные из таблиц.

- `SELECT DISTINCT` - чтобы данные не повторялись (выделяет все уникальные значения)
- `SELECT LIMIT n` - выбрать не более n записей

Порядок написания `SELECT`-запросов:
$$SELECT \rightarrow FROM \rightarrow WHERE \rightarrow GROUP\ BY \rightarrow HAVING \rightarrow ORDER\ BY \rightarrow LIMIT$$

Порядок выполнения `SELECT`-запросов:
$$FROM \rightarrow WHERE \rightarrow GROUP\ BY \rightarrow HAVING \rightarrow SELECT \rightarrow ORDER\ BY \rightarrow LIMIT$$

```sql
SELECT * FROM employees;
SELECT name, salary FROM employees WHERE salary > 50000;
```

  
##### `INSERT`

Добавляет новую строку в таблицу.

```sql
INSERT INTO employees (name, position, salary) VALUES ('Ivan', 'Manager', 60000);
```

  
##### `UPDATE`

Изменяет существующие данные.

```sql
UPDATE employees SET salary = salary * 1.1 WHERE position = 'Manager';
```


##### `DELETE`

Удаляет строки из таблицы.

```sql
DELETE FROM employees WHERE id = 1;
```

  
#### 3. [[Виды операторов в SQL#^bfa09f|DCL]] — Операторы управления доступом

##### `GRANT`

Предоставляет права пользователям.
```sql
GRANT SELECT, INSERT ON employees TO analyst;
```

  
##### `REVOKE`

Отзывает ранее выданные права.

```sql
REVOKE INSERT ON employees FROM analyst;
```

  
#### 4. [[Виды операторов в SQL#^b57371|TCL]] — Операторы управления [[Транзакции|транзакциями]]


| Оператор                    | Описание                                           | Пример использования              |
| --------------------------- | -------------------------------------------------- | --------------------------------- |
| **`BEGIN`**                 | Начинает блок транзакции                           | `BEGIN;` или `START TRANSACTION;` |
| **`COMMIT`**                | Фиксирует (применяет) транзакцию                   | `COMMIT;`                         |
| **`SAVEPOINT`**             | Создает точку сохранения в транзакции              | `SAVEPOINT backup_point;`         |
| **`ROLLBACK TO SAVEPOINT`** | Откатывает изменения до указанной точки сохранения | `ROLLBACK TO backup_point;`       |
| **`RELEASE SAVEPOINT`**     | Удаляет указанную точку сохранения                 | `RELEASE SAVEPOINT backup_point;` |
| **`ROLLBACK`**              | Откатывает всю текущую транзакцию                  | `ROLLBACK;`                       |

##### `BEGIN` / `START TRANSACTION`

Начинает транзакцию.

```sql
BEGIN;
```


##### `COMMIT`

Фиксирует изменения, сделанные в рамках транзакции.

```sql
COMMIT;
```


##### `ROLLBACK`

Откатывает транзакцию до последнего `BEGIN`.

```sql
ROLLBACK;
```

  
##### `SAVEPOINT`

Создаёт точку сохранения внутри транзакции.

```sql
SAVEPOINT sp1;
```

  
##### `ROLLBACK TO SAVEPOINT`

Откатывается к определённой точке внутри транзакции.

```sql
ROLLBACK TO SAVEPOINT sp1;
```
  

#### 5. Операторы условий и объединений

##### `WHERE`

Фильтрация строк по условию.

```sql
SELECT * FROM employees WHERE salary > 50000;
```


##### `GROUP BY`

^8fe7df

Группировка строк по значениям.

```sql
SELECT 
	position,
	avg(salary) 
FROM employees 
GROUP BY position;
```


##### `HAVING`

Фильтрация после группировки.

```sql
SELECT 
	position,
	avg(salary) 
FROM employees
GROUP BY position
HAVING AVG(salary) > 60000;
```


##### `ORDER BY`

Сортировка результатов.

  - `ASC` - сортировка по возрастанию
  - `DESC` - сортировка по убыванию

```sql
SELECT * FROM employees ORDER BY salary DESC;
```

  
#### 6. Прочие полезные операторы

##### `UNION`

^949bb1

Объединяет результаты нескольких запросов (уникальные строки).

```sql
SELECT name FROM clients

UNION

SELECT name FROM suppliers;
```


##### `INTERSECT`

Общие строки между запросами.

```sql
SELECT id FROM table1

INTERSECT

SELECT id FROM table2;
```

  
##### `EXCEPT`

Разница между двумя наборами данных.

```sql
SELECT id FROM table1

EXCEPT

SELECT id FROM table2;
```

#### 7. Объединения таблиц в SQL

##### `INNER JOIN`
Возвращает только те строки, у которых есть совпадения в обеих таблицах по условию соединения. Cтроки без совпадений исключаются из результата.

```sql
SELECT employees.name, departments.name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;
```


##### `LEFT JOIN`
Возвращает все строки из **левой** таблицы и совпадающие строки из правой. Если в правой таблице нет совпадения — будут `NULL`.

```sql
SELECT employees.name, departments.name
FROM employees
LEFT JOIN departments ON employees.department_id = departments.id;
```


##### `RIGHT JOIN`
Возвращает все строки из **правой** таблицы и совпадающие строки из левой. Если в левой таблице нет совпадения — будут `NULL`.

```sql
SELECT employees.name, departments.name
FROM employees
RIGHT JOIN departments ON employees.department_id = departments.id;
```


##### `FULL JOIN`

Возвращает все строки из обеих таблиц. Если совпадение есть — объединяет. Если нет — подставляет `NULL`.

```sql
SELECT employees.name, departments.name
FROM employees
FULL JOIN departments ON employees.department_id = departments.id;
```


##### `CROSS JOIN`

Декартово произведение: каждая строка из одной таблицы соединяется с каждой строкой другой.

```sql
SELECT e.name, d.name
FROM employees e
CROSS JOIN departments d;
```


##### `SELF JOIN`

Соединение таблицы с самой собой — через псевдонимы.

```sql
SELECT A.name AS employee, B.name AS manager
FROM employees A
JOIN employees B ON A.manager_id = B.id;
```


#### 8. Операторы сравнения

***Применяются в условиях (`WHERE`, `JOIN`, `CASE`, `HAVING` и т.п.).***

| Оператор       | Описание                                                       | Пример                                                                    |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `=`            | Равно                                                          | `salary = 50000`                                                          |
| `<>` или `!=`  | Не равно                                                       | `department_id != 3`                                                      |
| `>`            | Больше                                                         | `salary > 60000`                                                          |
| `<`            | Меньше                                                         | `salary < 40000`                                                          |
| `>=`           | Больше или равно                                               | `salary >= 50000`                                                         |
| `<=`           | Меньше или равно                                               | `hire_date <= '2023-01-01'`                                               |
| `BETWEEN`      | Диапазон значений                                              | `salary BETWEEN 40000 AND 60000`                                          |
| `IS NULL`      | Проверка на `NULL`                                             | `manager_id IS NULL`                                                      |
| `IS NOT NULL`  | Проверка на не `NULL`                                          | `email IS NOT NULL`                                                       |
| `IN`           | Проверка наличия в списке значений                             | `department_id IN (1, 2, 3)`                                              |
| `NOT IN`       | Проверка отсутствия в списке значений                          | `job_title NOT IN ('Manager', 'Intern')`                                  |
| `LIKE`         | Поиск по шаблону, чувствителен к регистру                      | `WHERE name LIKE 'A%'`                                                    |
| `ILIKE`        | (PostgreSQL) То же, что `LIKE`, но без учёта регистра          | `WHERE name ILIKE 'a%'`                                                   |
| `EXISTS`       | Проверка наличия хотя бы одного значения                       | `WHERE EXISTS (SELECT * FROM sales)`                                      |
| `NOT EXISTS`   | Проверка на пустоту или несуществование объекта                | `CREATE TABLE IF NOT EXISTS`                                              |
| `ALL`          | Проверяет чтобы все значения в множестве были истинны          | `WHERE price >= ALL (SELECT price FROM products WHERE price IS NOT NULL)` |
| `ANY`          | Проверяет чтобы хотя бы одно значение в множестве было истинно | `WHERE price >= ALL (SELECT price FROM products WHERE price IS NOT NULL)` |

Операторы `LIKE` и `ILIKE` работают с регулярными выражениями. Вот пару фактов о синтаксисе регулярок в SQL:
- `%` означает несколько (в том числе 0) символов;
- `_` означает ровно 1 символ.

#### 9. Логические операторы

| Оператор | Описание             |
| -------- | -------------------- |
| `AND`    | Логическое И         |
| `OR`     | Логическое ИЛИ       |
| `NOT`    | Логическое отрицание |
| `CASE`   | Условное выражение   |
Пример с `CASE`:

```sql
SELECT name,
       CASE
           WHEN salary > 80000 THEN 'High'
           WHEN salary BETWEEN 50000 AND 80000 THEN 'Medium'
           ELSE 'Low'
       END AS salary_level
FROM employees;
```

##### [[Тернарная логика]]

***В тернарной (троичной) логике к стандартным TRUE (T) и FALSE (F) добавляется UNKNOWN (U, он же NULL).***

- Если NULL как-то влияет на результат, то он становится не определён (UNKNOWN)
- Операторы `UNION`, `INTERSECT` и `EXCEPT` считают NULL за одинаковые значения (NULL = NULL), а оператор `UNION ALL` - за различные
- Оператор `DISTINCT` расценивает NULL как известное значение

| Исходное значение | `NOT` результат |
| ----------------- | --------------- |
| **FALSE**         | TRUE            |
| **UNKNOWN**       | UNKNOWN         |
| **TRUE**          | FALSE           |

| `AND`       | FALSE | UNKNOWN | TRUE    |
| ----------- | ----- | ------- | ------- |
| **FALSE**   | FALSE | FALSE   | FALSE   |
| **UNKNOWN** | FALSE | UNKNOWN | UNKNOWN |
| **TRUE**    | FALSE | UNKNOWN | TRUE    |

| `OR`        | FALSE   | UNKNOWN | TRUE |
| ----------- | ------- | ------- | ---- |
| **FALSE**   | FALSE   | UNKNOWN | TRUE |
| **UNKNOWN** | UNKNOWN | UNKNOWN | TRUE |
| **TRUE**    | TRUE    | TRUE    | TRUE |

| Оператор     | Условие                     | При наличии NULL в правой части                                                                |
| ------------ | --------------------------- | ---------------------------------------------------------------------------------------------- |
| **`IN`**     | `value IN (1, 2, NULL)`     | Возвращает: <br>• TRUE (если найдено совпадение) <br>• NULL (если совпадений нет и есть NULL)  |
| **`NOT IN`** | `value NOT IN (1, 2, NULL)` | Возвращает: <br>• FALSE (если найдено совпадение) <br>• NULL (если совпадений нет и есть NULL) |

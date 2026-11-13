В отличие от [[Value Iteration]], работающего в пространстве оценочных функций, алгоритм **Policy Iteration** ведёт поиск непосредственно в пространстве детерминированных [[Введение в обучение с подкреплением|стратегий]] $\pi\ :\ \mathcal{S} \to \mathcal{A}$, чередуя два шага:
* **Policy Evaluation** *(оценивание стратегии)* - нахождение точной функции ценности $V^{\pi_k}$ текущей стратегии $\pi_k$ путём решения [[Уравнение Беллмана#Точное нахождение функции ценности|СЛУ уравнений Беллмана]] $V^{\pi_k} = B^{\pi_k} V^{\pi_k}$.
* **Policy Improvement** *(улучшение стратегии)* - жадное обновление стратегии по найденной оценке:$$\pi_{k+1}(s) = \arg\max\limits_{a \in \mathcal{A}} \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V^{\pi_k}(s')$$

### Алгоритм Policy Iteration

Вход: [[Введение в обучение с подкреплением|MDP]] $(\mathcal{S}, \mathcal{A}, P_{sa}, \gamma, R)$ с фиксированным правилом разрешения ничьих в $\arg\max$.
1. Инициализировать детерминированную стратегию $\pi_0(s)$ произвольно $\forall s \in \mathcal{S}$, положить $k = 0$.
2. Повторять, пока $\exists s \in \mathcal{S}\ :\ \pi_k(s) \neq \pi_{k-1}(s)$:
	1. **Policy Evaluation**: найти $V^{\pi_k}$ как решение СЛУ из $|\mathcal{S}|$ линейных уравнений $\forall s \in \mathcal{S}$:$$V^{\pi_k}(s) - \gamma \sum\limits_{s' \in \mathcal{S}} P_{s\pi_k(s)}(s') V^{\pi_k}(s') = R(s)$$
	2. **Policy Improvement**: для каждого $s \in \mathcal{S}$ жадно обновить стратегию:$$\pi_{k+1}(s) = \arg\max\limits_{a \in \mathcal{A}} \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V^{\pi_k}(s')$$
	3. Увеличить счётчик итераций: $k \leftarrow k + 1$.

### Теорема о монотонном улучшении и конечной сходимости Policy Iteration

В конечном табличном MDP алгоритм Policy Iteration монотонно улучшает функцию ценности ($V^{\pi_{k+1}} \ge V^{\pi_k}$) и завершает работу за конечное число итераций, находя точную [[Value Iteration|оптимальную стратегию]] $\pi^*$.

#### Доказательство

1. **Монотонность шага**: пусть $\pi' = \pi_{k+1}$ получена жадно по $V^{\pi_k}$, тогда $\forall s \in \mathcal{S}$:$$V^{\pi_k}(s) = R(s) + \gamma \sum\limits_{s' \in \mathcal{S}} P_{s\pi_k(s)}(s') V^{\pi_k}(s') \le R(s) + \max\limits_{a \in \mathcal{A}} \gamma \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V^{\pi_k}(s') = [B^{\pi'} V^{\pi_k}](s)$$
2. Т.к. коэффициенты $\gamma P_{s\pi'(s)}(s') \ge 0$, [[Value Iteration|оператор Беллмана]] $B^{\pi'}$ монотонен. Применяя $B^{\pi'}$ к неравенству $V^{\pi_k} \le B^{\pi'} V^{\pi_k}$ рекурсивно $m$ раз и переходя к пределу при $m \to \infty$, получаем:$$V^{\pi_k} \le B^{\pi'} V^{\pi_k} \le (B^{\pi'})^2 V^{\pi_k} \le \dots \le \lim\limits_{m \to \infty} (B^{\pi'})^m V^{\pi_k} = V^{\pi'}$$
3. **Критерий останова**: если $\pi_{k+1} \equiv \pi_k$, то $V^{\pi_k} = B^{\pi_{k+1}} V^{\pi_k} = B V^{\pi_k}$, т.е. $V^{\pi_k}$ удовлетворяет уравнению оптимальности Беллмана и в силу единственности неподвижной точки $V^{\pi_k} \equiv V^* \implies \pi_k \equiv \pi^*$.
4. **Конечность числа шагов**: пока остановка не произошла, $\pi_{k+1} \neq \pi_k$, что при фиксированном правиле разрешения ничьих в $\arg\max$ влечёт строгое неравенство $V^{\pi_{k+1}}(s) > V^{\pi_k}(s)$ хотя бы в одном состоянии $s \in \mathcal{S}$, т.ч. стратегии в цепочке $\pi_0, \pi_1, \dots, \pi_k$ не повторяются. Т.к. общее число детерминированных стратегий $|\mathcal{A}|^{|\mathcal{S}|} < \infty$, алгоритм завершится не более чем за $|\mathcal{A}|^{|\mathcal{S}|}$ итераций.

#### Асимптотика

1. Сложность одной итерации по времени: $O(|\mathcal{S}|^3 + |\mathcal{S}|^2 |\mathcal{A}|)$, где $O(|\mathcal{S}|^3)$ тратится на точное решение СЛУ размера $|\mathcal{S}| \times |\mathcal{S}|$ на шаге Policy Evaluation, а $O(|\mathcal{S}|^2 |\mathcal{A}|)$ - на шаг Policy Improvement.
2. Число итераций: в худшем случае $O(|\mathcal{A}|^{|\mathcal{S}|})$, на практике сходится за малое число шагов.
3. Сложность по памяти: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ - хранение тензора переходов $P_{sa}(s')$ и матрицы системы $(E - \gamma P_{s\pi(s)}) \in Mat_{|\mathcal{S}| \times |\mathcal{S}|}(\mathbb{R})$.

### Generalized Policy Iteration (GPI)

==Опр.== **Generalized Policy Iteration** *(GPI, обобщённая итерация по стратегии)* - схема динамического программирования, в которой точное решение СЛУ на этапе Policy Evaluation заменяется $N \in \mathbb{N}$ итерациями применения оператора $B^\pi$:
1. Инициализировать $V(s) = 0$ и $\pi(s)$ произвольно $\forall s \in \mathcal{S}$.
2. Повторять до сходимости:
	1. **Частичный Policy Evaluation**: повторить $N$ раз обновление $\forall s \in \mathcal{S}\ :\ V(s) \leftarrow R(s) + \gamma \sum\limits_{s' \in \mathcal{S}} P_{s\pi(s)}(s') V(s')$.
	2. **Policy Improvement**: обновить стратегию $\forall s \in \mathcal{S}\ :\ \pi(s) \leftarrow \arg\max\limits_{a \in \mathcal{A}} \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V(s')$.

> [!IMPORTANT] Связь алгоритмов через GPI
> При $N = \infty$ схема GPI переходит в классический **Policy Iteration**, а при $N = 1$ вырождается в **[[Value Iteration]]**. Парадигма чередования оценивания *(критик)* и улучшения *(актёр)* лежит в основе архитектур Actor-Critic.

#### Сведение GPI к Value Iteration при $N = 1$

==Утв.== Одна итерация алгоритма GPI при $N = 1$ эквивалентна одному синхронному шагу алгоритма Value Iteration.

##### Доказательство

1. При $N = 1$ шаг оценивания в GPI выполняет одно обновление для текущей жадной стратегии $\pi(s) = \arg\max\limits_{a \in \mathcal{A}} \left( R(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V(s') \right)$:$$V_{\text{new}}(s) = R(s, \pi(s)) + \gamma \sum\limits_{s' \in \mathcal{S}} P_{s\pi(s)}(s') V(s')$$
2. Подставляя выражение для $\pi(s)$ в $V_{\text{new}}(s)$, получаем оператор оптимальности Беллмана:$$V_{\text{new}}(s) = \max\limits_{a \in \mathcal{A}} \left( R(s, a) + \gamma \sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V(s') \right) = [BV](s)$$

### Сравнительный анализ методов динамического программирования

| Критерий сравнения | [[Value Iteration]] | Policy Iteration | Generalized Policy Iteration (GPI) |
| :---: | :---: | :---: | :---: |
| **Пространство итераций** | Пространство функций $\mathbb{R}^{\vert\mathcal{S}\vert}$ | Пространство стратегий $\mathcal{A}^{\vert\mathcal{S}\vert}$ | Совместно $(V, \pi)$ |
| **Сложность 1 итерации** | $O(\vert\mathcal{S}\vert^2 \vert\mathcal{A}\vert)$ | $O(\vert\mathcal{S}\vert^3 + \vert\mathcal{S}\vert^2 \vert\mathcal{A}\vert)$ | $O(N \vert\mathcal{S}\vert^2 \vert\mathcal{A}\vert)$ |
| **Скорость сходимости** | Геометрическая со знаменателем $\gamma$ | Конечное число шагов | Промежуточная (сжатие оценивания $\gamma^N$) |
| **Явная стратегия на шаге $k$** | Отсутствует (вычисляется в конце) | Присутствует на каждой итерации | Присутствует на каждой итерации |
| **Масштаб $\vert\mathcal{S}\vert$** | Большие табличные пространства | Малые пространства состояний | Произвольные табличные пространства |

### +ы & -ы методов динамического программирования

> [!SUCCESS] Преимущества
> * Гарантированная сходимость к глобально оптимальной стратегии $\pi^*$ и функции ценности $V^*$ вне зависимости от начальной инициализации.
> * Отсутствие дисперсии сэмплирования: на каждом шаге вычисляется точное математическое ожидание по всем переходам $s' \in \mathcal{S}$.

> [!FAILURE] Недостатки
> * Требование полного знания модели среды *(model-based)*: необходимы явные вероятности переходов $P_{sa}(s')$ и функция наград $R(s)$.
> * Проклятие размерности: кубическая сложность $O(|\mathcal{S}|^3)$ у Policy Iteration и квадратичная зависимость $O(|\mathcal{S}|^2 |\mathcal{A}|)$ у Value Iteration делают табличные методы неприменимыми при огромных или непрерывных пространствах состояний, требуя перехода к [[SARSA и Q-learning|TD-методам]] и [[Policy Gradient]].

---

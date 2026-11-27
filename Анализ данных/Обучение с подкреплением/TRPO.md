В методах градиента стратегии *(REINFORCE, [[Advantage и Actor-Critic|A2C]])* остаётся проблема слишком большого шага по политике:
1. **Чувствительность пространства политик к шагу в параметрах**: фиксированный шаг $\alpha$ в евклидовом пространстве весов $\theta \in \mathbb{R}^d$ может соответствовать радикальному сдвигу распределения действий $\pi_\theta(a \mid s)$.
2. **Policy collapse**: в отличие от обучения с учителем, в RL политика сама собирает данные для следующей итерации. Один неудачный шаг ухудшает $\pi_\theta \implies$ агент собирает плохие траектории $\implies$ дальнейшие градиенты ведут в ещё худшие области.

Цель - контролировать изменение самой политики в метрике KL-дивергенции, а не в евклидовой метрике параметров $\theta$.

==Опр.== **Суррогатный функционал** - локальная аппроксимация прироста ожидаемой награды по сэмплам старой политики $\pi_{\theta_{\text{old}}}$ с поправкой на смену распределения:$$L_{\theta_{\text{old}}}(\theta) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}},\, a \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \, A^{\pi_{\text{old}}}(s, a) \right]$$
==Опр.== **Доверительная область** *(trust region)* - окрестность текущей политики $\theta_{\text{old}}$, в которой средняя дивергенция Кульбака-Лейблера между старым и новым распределениями действий не превышает $\delta > 0$:$$\overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta) = \mathbb{E}_{s \sim d^{\pi_{\text{old}}}} \mathrm{KL}\left(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\right) \le \delta$$
==Опр.== **Задача TRPO** *(Trust Region Policy Optimization)* - условная максимизация суррогатного функционала внутри доверительной области:$$\sup\limits_{\theta \in \mathbb{R}^d} \{L_{\theta_{\text{old}}}(\theta)\ |\ \overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta) \le \delta\}$$
Вводя шаг $\Delta = \theta - \theta_{\text{old}}$ и применяя [[Формула Тейлора для функций многих переменных|разложение Тейлора]] в окрестности $\theta_{\text{old}}$ *(1-го порядка для цели и 2-го порядка для ограничения, т.к. 0-й и 1-й члены разложения KL равны нулю)*, получаем приближённую задачу:$$\sup\limits_{\Delta \in \mathbb{R}^d} \{g^\top \Delta\ |\ \frac{1}{2} \Delta^\top H \Delta \le \delta\}$$где $g = \nabla_\theta L_{\theta_{\text{old}}}(\theta)\big|_{\theta=\theta_{\text{old}}}$ - градиент политики, $H = \nabla_\theta^2 \overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta)\big|_{\theta=\theta_{\text{old}}}$

> [!INFO] Геометрический смысл: натуральный градиент
> Обычный градиентный подъём $\theta_{\text{new}} = \theta_{\text{old}} + \alpha g$ ищет максимум линейной аппроксимации внутри евклидова шара $\|\Delta\|_2^2 \le \varepsilon$. Ограничение $\frac{1}{2} \Delta^\top H \Delta \le \delta$ задаёт эллипсоид в пространстве параметров: вдоль направлений высокой кривизны шаг сжимается, вдоль «плоских» - удлиняется. Направление $H^{-1} g$ - **натуральный градиент** *(natural policy gradient)*, задающий наискорейший подъём в локальной метрике KL-дивергенции.
> ![[TRPO_natural_gradient.png]]

> [!TIP] Зачем нужен суррогат и почему нельзя уходить далеко от $\theta_{\text{old}}$
> Суррогат $L_{\theta_{\text{old}}}(\theta)$ вычисляется полностью по уже собранным данным старой политики, а его градиент в точке $\theta = \theta_{\text{old}}$ пропорционален истинному градиенту $J(\theta)$. При удалении от $\theta_{\text{old}}$ распределение состояний $d^{\pi_\theta}$ начинает отличаться от $d^{\pi_{\text{old}}}$ и аппроксимация ломается - ограничение $\overline{\mathrm{KL}} \le \delta$ удерживает шаг в области применимости суррогата.

> [!IMPORTANT] Порядок аргументов в KL-дивергенции
> В TRPO используется прямой порядок $\mathrm{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta)$: матожидание берётся по $\pi_{\theta_{\text{old}}}$, т.е. по уже собранным действиям батча. В точке $\theta = \theta_{\text{old}}$ гессианы обоих порядков совпадают, но при проверке точного ограничения на этапе line search порядок существенен.

> [!EXAMPLE] Пример: **Подъём в тумане**
> При движении в гору в густом тумане рельеф виден лишь в радиусе $\delta$ *(доверительная область)*. Вместо фиксированного шага строится локальная карта уклона ($g$) и кривизны ($H$), вычисляется оптимальный шаг до границы видимости, а затем проверяется реальный рельеф *(backtracking line search)* - при нарушении условий шаг уменьшается вдвое.

> [!WARNING] Матрица $H$ не материализуется
> Для нейросети с $d$ параметрами матрица $H \in Mat_{d \times d}(\mathbb{R})$ требует $O(d^2)$ памяти и не может быть построена или обращена. Систему $H x = g$ решают приближённо методом **сопряжённых градиентов**, которому нужен только оператор умножения гессиана на вектор $v \mapsto H v$, вычисляемый за два [[Автоматическое дифференцирование|автодифференцирования]]: $H v = \nabla_\theta \left( (\nabla_\theta \overline{\mathrm{KL}})^\top v \right)$.

### Локальное совпадение градиентов суррогата и целевого функционала
$$\nabla_\theta L_{\theta_{\text{old}}}(\theta)\Big|_{\theta=\theta_{\text{old}}} = (1 - \gamma) \, \nabla_\theta J(\theta)\Big|_{\theta=\theta_{\text{old}}}$$
#### Доказательство

1. Распределения $d^{\pi_{\text{old}}}$, $\pi_{\theta_{\text{old}}}$ и $A^{\pi_{\text{old}}}$ не зависят от $\theta$. Дифференцируя отношение вероятностей в точке $\theta = \theta_{\text{old}}$:$$\nabla_\theta \left( \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \right) \Bigg|_{\theta=\theta_{\text{old}}} = \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \Bigg|_{\theta=\theta_{\text{old}}} = \nabla_\theta \log \pi_\theta(a \mid s) \Big|_{\theta=\theta_{\text{old}}}$$
2. Внося градиент под матожидание, получаем:$$\nabla_\theta L_{\theta_{\text{old}}}(\theta)\Big|_{\theta=\theta_{\text{old}}} = \mathbb{E}_{s \sim d^{\pi_{\text{old}}},\, a \sim \pi_{\theta_{\text{old}}}} \left[ \nabla_\theta \log \pi_\theta(a \mid s)\Big|_{\theta=\theta_{\text{old}}} \cdot A^{\pi_{\text{old}}}(s, a) \right]$$
3. По представлению градиента стратегии через $d^{\pi_{\text{old}}}$ и $A^{\pi_{\text{old}}}$ правая часть в точности равна $(1 - \gamma) \nabla_\theta J(\theta)\big|_{\theta=\theta_{\text{old}}}$.

### Совпадение гессиана KL-дивергенции с матрицей Фишера

В точке $\theta = \theta_{\text{old}}$ выполняется $\overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta_{\text{old}}) = 0$, $\nabla_\theta \overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta)\big|_{\theta=\theta_{\text{old}}} = 0$, а гессиан равен матрице информации Фишера:$$H = \nabla_\theta^2 \overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta)\Big|_{\theta=\theta_{\text{old}}} = \mathbb{E}_{s \sim d^{\pi_{\text{old}}},\, a \sim \pi_{\theta_{\text{old}}}} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, \nabla_\theta \log \pi_\theta(a \mid s)^\top \right]\Bigg|_{\theta=\theta_{\text{old}}} \succeq 0$$
#### Доказательство

1. При фиксированном $s \in \mathcal{S}$ имеем $\mathrm{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}} \left[\log \pi_{\theta_{\text{old}}}(a \mid s) - \log \pi_\theta(a \mid s)\right]$. При $\theta = \theta_{\text{old}}$ разность равна $0 \implies \overline{\mathrm{KL}}(\theta_{\text{old}} \| \theta_{\text{old}}) = 0$.
2. По Л. о нулевом матожидании градиента логарифма стратегии первая производная зануляется:$$\nabla_\theta \mathrm{KL}\big|_{\theta=\theta_{\text{old}}} = -\mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}} \nabla_\theta \log \pi_\theta(a \mid s)\Big|_{\theta=\theta_{\text{old}}} = 0$$
3. Вычислим матрицу вторых производных от $-\log \pi_\theta(a \mid s)$:$$\nabla_\theta^2 (-\log \pi_\theta) = -\nabla_\theta \left( \frac{\nabla_\theta \pi_\theta}{\pi_\theta} \right) = -\frac{\nabla_\theta^2 \pi_\theta}{\pi_\theta} + \frac{\nabla_\theta \pi_\theta \, \nabla_\theta \pi_\theta^\top}{\pi_\theta^2} = -\frac{\nabla_\theta^2 \pi_\theta}{\pi_\theta} + \nabla_\theta \log \pi_\theta \, \nabla_\theta \log \pi_\theta^\top$$
4. Взяв матожидание по $a \sim \pi_{\theta_{\text{old}}}(\cdot \mid s)$ в точке $\theta = \theta_{\text{old}}$, получаем зануление первого слагаемого:$$\mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\nabla_\theta^2 \pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \right]\Bigg|_{\theta=\theta_{\text{old}}} = \int\limits_{\mathcal{A}} \nabla_\theta^2 \pi_\theta(a \mid s) \, da = \nabla_\theta^2 \int\limits_{\mathcal{A}} \pi_\theta(a \mid s) \, da = \nabla_\theta^2 1 = 0$$
5. Усредняя оставшееся второе слагаемое по $s \sim d^{\pi_{\text{old}}}$, получаем $H = \mathbb{E}_{s, a} [v v^\top]$, где $v = \nabla_\theta \log \pi_\theta$. Т.к. $\forall u \in \mathbb{R}^d\ :\ u^\top (v v^\top) u = (u^\top v)^2 \ge 0$, матрица $H$ симметрична и неотрицательно определена ($H \succeq 0$).

### Решение приближённой задачи TRPO

Пусть $H \succ 0$, $g \neq 0$, $\delta > 0$. Тогда единственное решение задачи $\sup\limits_{\Delta \in \mathbb{R}^d} g^\top \Delta$ при условии $\frac{1}{2} \Delta^\top H \Delta \le \delta$ имеет вид:$$\Delta^* = \sqrt{\frac{2\delta}{g^\top H^{-1} g}} \, H^{-1} g$$
#### Доказательство

1. Целевая функция $g^\top \Delta$ линейна *(вогнута)*, ограничение $\frac{1}{2} \Delta^\top H \Delta - \delta \le 0$ строго выпукло ($H \succ 0$), условие Слейтера выполнено *(при $\Delta = 0\ :\ -\delta < 0$)*. Запишем лагранжиан при $\lambda \ge 0$:$$\mathcal{L}(\Delta, \lambda) = g^\top \Delta - \lambda \left( \frac{1}{2} \Delta^\top H \Delta - \delta \right)$$
2. Из [[Теорема Куна-Таккера|условий ККТ]] имеем $\nabla_\Delta \mathcal{L}(\Delta, \lambda) = g - \lambda H \Delta = 0$. Случай $\lambda = 0$ невозможен, т.к. $g \neq 0 \implies \lambda > 0$. По условию дополняющей нежёсткости ограничение активно ($\frac{1}{2} (\Delta^*)^\top H \Delta^* = \delta$), а шаг выражается как:$$\Delta^* = \frac{1}{\lambda} H^{-1} g$$
3. Подставляя $\Delta^*$ в активное ограничение, находим множитель $\lambda$:$$\frac{1}{2 \lambda^2} (H^{-1} g)^\top H (H^{-1} g) = \frac{1}{2 \lambda^2} g^\top H^{-1} g = \delta \iff \frac{1}{\lambda} = \sqrt{\frac{2\delta}{g^\top H^{-1} g}}$$
4. Подстановка $\frac{1}{\lambda}$ в выражение для $\Delta^*$ даёт искомый шаг.

### Алгоритм TRPO (с CG и Backtracking Line Search)

Вход: начальные параметры $\theta_0, \phi_0$, радиус доверительной области $\delta > 0$, коэффициент бэктрэкинга $\alpha \in (0; 1)$, число шагов бэктрэкинга $K$, число итераций CG $M_{\text{CG}}$.
1. Для каждой итерации $k = 0, 1, 2, \dots$:
	1. **Сбор данных**: запустить текущую политику $\pi_k = \pi_{\theta_k}$ и собрать батч траекторий $\mathcal{D}_k = \{\tau_i\}$.
	2. **Оценка Advantage**: вычислить возвраты $\hat{R}_t$ и оценки $\hat{A}_t$ *(через GAE или MC)* с помощью критика $V_{\phi_k}$.
	3. **Оценка градиента суррогата**:$$\hat{g}_k = \frac{1}{|\mathcal{D}_k| T} \sum\limits_{\tau \in \mathcal{D}_k} \sum\limits_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big|_{\theta_k} \cdot \hat{A}_t$$
	4. **Поиск направления**: за $M_{\text{CG}}$ итераций метода сопряжённых градиентов *(с демпфированием $\hat{H}_k + \xi E$)* найти приближённое решение системы $\hat{H}_k x = \hat{g}_k$, т.ч. $\hat{x}_k \approx \hat{H}_k^{-1} \hat{g}_k$.
	5. **Расчёт максимального шага**:$$\Delta_{\max} = \sqrt{\frac{2\delta}{\hat{x}_k^\top \hat{H}_k \hat{x}_k}} \, \hat{x}_k$$
	6. **Backtracking line search**: найти наименьшее $j \in \overline{0, K}$, при котором для $\theta' = \theta_k + \alpha^j \Delta_{\max}$ выполнены нелинеаризованные условия:$$\hat{L}_{\theta_k}(\theta') > \hat{L}_{\theta_k}(\theta_k) \quad \wedge \quad \widehat{\mathrm{KL}}(\theta_k \| \theta') \le \delta$$Положить $\theta_{k+1} = \theta'$ *(если подходящего $j$ нет, оставить $\theta_{k+1} = \theta_k$)*.
	7. **Обновление критика**: минимизировать MSE по собранным данным:$$\phi_{k+1} = \arg\inf\limits_\phi \frac{1}{|\mathcal{D}_k| T} \sum\limits_{\tau \in \mathcal{D}_k} \sum\limits_{t=0}^{T-1} \left( V_\phi(s_t) - \hat{R}_t \right)^2$$
#### Асимптотика

Для одной итерации обновления актора по батчу из $B = |\mathcal{D}_k| T$ шагов среды при $d = |\theta|$ параметрах:
1. Сложность по времени:
	* Наивное обращение $H$: $O(B \cdot C_\pi + d^3)$ - неприменимо для нейросетей.
	* С оптимизациями приближений решений: $O((1 + M_{\text{CG}} + K) \cdot B \cdot C_\pi + M_{\text{CG}} \cdot d)$, где $C_\pi$ - стоимость прохода через $\pi_\theta$.
2. Сложность по памяти: $O(B \cdot \dim \mathcal{S} + d + |\phi|)$.

### Сравнительный анализ A2C и TRPO

|        Характеристика         |             A2C             |                                        TRPO                                         |
| :---------------------------: | :-------------------------: | :---------------------------------------------------------------------------------: |
|     **Ограничение шага**      | Нет *(только шаг $\alpha$)* | Жёсткое: $\overline{\mathrm{KL}}(\pi_{\text{old}} \parallel \pi_\theta) \le \delta$ |
|        **Line search**        |             Нет             |                                     Обязателен                                      |
| **Общий backbone с критиком** |             Да              |                                      Затруднён                                      |
|   **Сложность реализации**    |           Низкая            |                                       Высокая                                       |

### +ы & -ы TRPO

> [!SUCCESS] Преимущества
> * Теоретически обоснованное монотонное улучшение политики *(в рамках нижней оценки)* и устойчивость к policy collapse.
> * Инвариантность шага к масштабированию параметров благодаря метрике KL *(натуральный градиент)*.
> * Радиус $\delta$ имеет ясный смысл и менее чувствителен к настройке, чем learning rate в A2C.

> [!FAILURE] Недостатки
> * Вычислительно тяжёлый и сложный в реализации.
> * Требует крупных батчей $B$ для оценки $H$ и делает ровно одно обновление за сбор данных *(данные не переиспользуются)*.
> * Плохо сочетается с общим backbone, dropout и батч-нормализацией, плохо масштабируется на сверхбольшие модели.

---

Метод [[TRPO]] решает проблему разрушительно больших шагов за счёт жёсткой доверительной области по KL-дивергенции, но на практике громоздок: требует вычислений второго порядка *(HVP, CG, backtracking line search)*, делает лишь одно обновление за сбор батча и плохо сочетается с общим backbone и стандартными оптимизаторами первого порядка.

Цель **PPO** *(Proximal Policy Optimization)* - сохранить устойчивость доверительной области, перейдя к оптимизации первого порядка *(SGD / [[Оптимизации градиентного спуска|Adam]])* с многоэпоховым обновлением по одному собранному батчу.

==Опр.== **Отношение правдоподобия стратегий** *(probability ratio)* - относительное изменение вероятности выбранного действия $a_t$ в состоянии $s_t$ при обновлении параметров:$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
При $\theta = \theta_{\text{old}}$ выполняется $r_t(\theta_{\text{old}}) = 1$, а суррогатный функционал имеет вид $L^{\text{CPI}}(\theta) = \hat{\mathbb{E}}_t [r_t(\theta) \hat{A}_t]$. Для сдерживания шага предложены две базовые версии функционала актора:

==Опр.== **PPO-Penalty** *(PPO с адаптивным KL-штрафом)* - версия PPO, заменяющая жёсткое ограничение TRPO на мягкий KL-штраф с динамическим коэффициентом $\beta > 0$:$$L^{\text{KLPEN}}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \, \mathrm{KL}\left( \pi_{\theta_{\text{old}}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t) \right) \right]$$После обновления вычисляется выборочная дивергенция $d = \hat{\mathbb{E}}_t \mathrm{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \,\|\, \pi_\theta(\cdot \mid s_t))$ и сравнивается с порогом $d_{\text{targ}}$:$$\beta \leftarrow \begin{cases} \beta / 2, & d < d_{\text{targ}} / 1.5 \\ 2\beta, & d > 1.5 \, d_{\text{targ}} \\ \beta, & \text{иначе} \end{cases}$$
==Опр.== **PPO-Clip** *(PPO с клиппированным суррогатом)* - основная практическая версия PPO, ограничивающая стимул выводить $r_t(\theta)$ за пределы $[1 - \varepsilon; 1 + \varepsilon]$ с помощью отсечения *(clipping)* и пессимистичного минимума при $\varepsilon \in (0; 1)$:$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\Big( r_t(\theta) \hat{A}_t,\; \operatorname{clip}\big(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon\big) \hat{A}_t \Big) \right]$$где $\operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon) = \max(1 - \varepsilon, \min(r, 1 + \varepsilon))$.

==Опр.== **Полный функционал PPO** - объединённый функционал для совместного обучения актора и критика *(в т.ч. при общем backbone)*, максимизируемый градиентным подъёмом:$$L^{\text{PPO}}(\theta, \phi) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 \left( V_\phi(s_t) - V_t^{\text{targ}} \right)^2 + c_2 \, H\big(\pi_\theta(\cdot \mid s_t)\big) \right]$$где $V_t^{\text{targ}} = \hat{A}_t^{\text{GAE}} + V_{\phi_{\text{old}}}(s_t)$, $H(\pi_\theta)$ - энтропия стратегии, $c_1, c_2 > 0$.

> [!TIP] Интуиция PPO-Clip: зачем нужен $\min$
> PPO-Clip не запрещает политике менять вероятности механически, а убирает награду за чрезмерный сдвиг:
> * При $\hat{A}_t > 0$ и $r_t(\theta) \ge 1 + \varepsilon$ функционал выходит на плато $(1 + \varepsilon)\hat{A}_t$ и градиент по этому сэмплу обнуляется.
> * При $\hat{A}_t < 0$ и $r_t(\theta) \le 1 - \varepsilon$ функционал выходит на плато $(1 - \varepsilon)\hat{A}_t$ и дальнейшее снижение не даёт выигрыша.
> * **Роль $\min$**: если политика качнулась в невыгодную сторону *(например, $\hat{A}_t > 0$, но $r_t < 1 - \varepsilon$)*, оператор $\min$ выбирает неклиппированное слагаемое $r_t \hat{A}_t$, сохраняя ненулевой градиент для возврата в безопасную зону.

> [!INFO] Почему PPO допускает несколько эпох по одному батчу
> В обычном PG многократный проход по батчу уводит $\pi_\theta$ далеко от $\pi_{\theta_{\text{old}}}$, где данные уже невалидны. В PPO-Clip как только для пары $(s_t, a_t)$ отношение $r_t$ достигает границы $1 \pm \varepsilon$, сэмпл выключается из градиента. Ограниченный бюджет влияния каждого примера позволяет безопасно выполнять $K \in \overline{3, 10}$ эпох по мини-батчам и повышает sample efficiency.

> [!EXAMPLE] Пример: **Поводок длины $\varepsilon$**
> * **TRPO** - жёсткий забор вокруг $\pi_{\theta_{\text{old}}}$: шаг физически отбрасывается *(line search)* при пересечении $\delta$.
> * **PPO-Penalty** - эластичный поводок: чем дальше уходит политика, тем сильнее тянет назад KL-штраф с подстраиваемой жёсткостью $\beta$.
> * **PPO-Clip** - поводок длины $\varepsilon$: за пределами $[1 - \varepsilon; 1 + \varepsilon]$ поощрение обнуляется *(мягкий trust region, т.к. градиенты других сэмплов батча всё же могут слегка вывести отдельное $r_t$ за границу)*.

### Кусочная структура и обнуление градиента в PPO-Clip

Пусть $\ell(r; A) = \min\big(r A,\; \operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon) A\big)$ для $r > 0$, $A \in \mathbb{R}$, $\varepsilon \in (0; 1)$. Тогда:
1. $\forall r > 0\ \forall A \in \mathbb{R}\ :\ \ell(r; A) \le r A$ *(пессимистичная нижняя оценка суррогата)*.
2. При $A > 0$ и $r > 1 + \varepsilon\ :\ \ell(r; A) = (1 + \varepsilon) A$ и $\nabla_\theta \ell(r(\theta); A) = 0$.
3. При $A < 0$ и $r < 1 - \varepsilon\ :\ \ell(r; A) = (1 - \varepsilon) A$ и $\nabla_\theta \ell(r(\theta); A) = 0$.
4. В невыгодных областях ($A > 0, r < 1 - \varepsilon$ или $A < 0, r > 1 + \varepsilon$): $\ell(r; A) = r A$ и $\nabla_\theta \ell(r(\theta); A) = A \nabla_\theta r(\theta)$.

#### Доказательство

1. Первое свойство следует из определения минимума: $\min(x, y) \le x \implies \ell(r; A) \le r A$.
2. **Случай $A > 0$**: умножение на $A > 0$ сохраняет порядок $\implies \ell(r; A) = \min\big(r, \operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon)\big) \cdot A$.
	* При $r \le 1 - \varepsilon\ :\ \operatorname{clip}(r) = 1 - \varepsilon \ge r \implies \ell(r; A) = r A$ *(градиент тянет $r$ вверх)*.
	* При $1 - \varepsilon < r < 1 + \varepsilon\ :\ \operatorname{clip}(r) = r \implies \ell(r; A) = r A$.
	* При $r \ge 1 + \varepsilon\ :\ \operatorname{clip}(r) = 1 + \varepsilon \le r \implies \ell(r; A) = (1 + \varepsilon) A = \mathrm{const}(\theta) \implies \nabla_\theta \ell = 0$.
3. **Случай $A < 0$**: умножение на $A < 0$ обращает порядок $\implies \ell(r; A) = \max\big(r, \operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon)\big) \cdot A$.
	* При $r \le 1 - \varepsilon\ :\ \operatorname{clip}(r) = 1 - \varepsilon \ge r \implies \ell(r; A) = (1 - \varepsilon) A = \mathrm{const}(\theta) \implies \nabla_\theta \ell = 0$.
	* При $1 - \varepsilon < r < 1 + \varepsilon\ :\ \operatorname{clip}(r) = r \implies \ell(r; A) = r A$.
	* При $r \ge 1 + \varepsilon\ :\ \operatorname{clip}(r) = 1 + \varepsilon \le r \implies \ell(r; A) = r A$ *(градиент тянет $r$ вниз)*.

### Локальное совпадение градиента PPO-Clip с PG
$$\nabla_\theta L^{\text{CLIP}}(\theta)\Big|_{\theta=\theta_{\text{old}}} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big|_{\theta=\theta_{\text{old}}} \cdot \hat{A}_t \right]$$
#### Доказательство

1. В точке $\theta = \theta_{\text{old}}$ для всех сэмплов батча $r_t(\theta_{\text{old}}) = 1$.
2. Т.к. $\varepsilon > 0$, точка $1 \in (1 - \varepsilon; 1 + \varepsilon)$, и по непрерывности $\pi_\theta$ $\exists$ окрестность $U(\theta_{\text{old}})$, в которой для всех сэмплов батча $\forall \theta \in U\ :\ r_t(\theta) \in (1 - \varepsilon; 1 + \varepsilon)$ и $\ell(r_t(\theta); \hat{A}_t) = r_t(\theta) \hat{A}_t$.
3. Дифференцируя в точке $\theta = \theta_{\text{old}}$:$$\nabla_\theta r_t(\theta)\Big|_{\theta=\theta_{\text{old}}} = \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\Bigg|_{\theta=\theta_{\text{old}}} = \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big|_{\theta=\theta_{\text{old}}}$$откуда следует утверждение.

### Законность нормировки Advantage по батчу

При фиксированных на шаге $\mu \in \mathbb{R}$, $\sigma > 0$ нормировка $\tilde{A}_t = \frac{\hat{A}_t - \mu}{\sigma}$ сохраняет направление математического ожидания градиента стратегии:$$\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \frac{\hat{A}_t - \mu}{\sigma} \right] = \frac{1}{\sigma} \, \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right]$$
#### Доказательство

1. Константа $\mu$ не зависит от $a_t$, т.ч. по Т. о baseline-инвариантности $\mathbb{E}[\nabla_\theta \log \pi_\theta \cdot (\hat{A}_t - \mu)] = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot \hat{A}_t]$.
2. Вынесение $\frac{1}{\sigma} > 0$ за знак матожидания лишь масштабирует длину градиента *(эквивалентно адаптивному learning rate, инвариантному к масштабу наград)*.

### Алгоритм PPO-Clip

Вход: $M$ параллельных сред, длина роллаута $N$, параметры $\theta_0, \phi_0$, порог $\varepsilon \in (0; 1)$, веса $c_1, c_2 > 0$, число эпох $K$, размер мини-батча $B \le MN$, параметры GAE $(\gamma, \lambda)$, опциональный порог $\mathrm{KL}_{\max}$.
1. Для каждой внешней итерации $k = 0, 1, 2, \dots$:
	1. Зафиксировать веса старой политики $\theta_{\text{old}} \leftarrow \theta_k$.
	2. **Сбор роллаута**: выполнить по $N$ шагов в $M$ средах под управлением $\pi_{\theta_{\text{old}}}$, сохраняя датасет $\mathcal{D}$ из $M \cdot N$ переходов $(s_t, a_t, R_t, s_{t+1})$, старые лог-вероятности $\log \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ и оценки $V_{\phi_k}(s_t)$.
	3. **Оценка Advantage и таргетов критика**:
		1. Обратным проходом вычислить GAE-оценки $\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}$ и таргеты $V_t^{\text{targ}} = \hat{A}_t + V_{\phi_k}(s_t)$ с фиксацией `detach`.
		2. Нормировать оценки по батчу:$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$$
	4. **Многоэпоховое обновление (Adam)**: для каждой эпохи $e \in \overline{1, K}$ перемешать $\mathcal{D}$, разбить на мини-батчи размера $B$ и для каждого мини-батча:
		1. Вычислить отношение вероятностей и полный функционал:$$r_t(\theta) = \exp\big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\big)$$$$L^{\text{PPO}}(\theta, \phi) = \frac{1}{B} \sum\limits_{t \in \text{batch}} \left[ \min\big(r_t(\theta)\hat{A}_t,\, \operatorname{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon)\hat{A}_t\big) - c_1 \big(V_\phi(s_t) - V_t^{\text{targ}}\big)^2 + c_2 H\big(\pi_\theta(\cdot \mid s_t)\big) \right]$$
		2. Выполнить шаг Adam на максимизацию $L^{\text{PPO}}(\theta, \phi)$ *(опционально с клиппингом нормы градиента)*.
		3. *(Опционально)* При $\widehat{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) > \mathrm{KL}_{\max}$ досрочно прервать цикл по эпохам *(early stopping)*.

#### Асимптотика

Для одной внешней итерации при объёме роллаута $D = M \cdot N$, числе эпох $K$, стоимости прохода через сеть $C_{\text{net}}$ и числе параметров $d = |\theta| + |\phi|$:
1. Сложность по времени: $O(D \cdot C_{\text{fwd}} + K \cdot D \cdot C_{\text{net}} + K \cdot \frac{D}{B} \cdot d)$ - без HVP и CG, каждый шаг стоит как обычный шаг SGD.
2. Сложность по памяти:
	* Классический PPO: $O(D \cdot \dim \mathcal{S} + d)$ - буфер роллаута, веса и состояния Adam.
	* RLHF-PPO для LLM размера $P$: $O(4P)$ на веса моделей ($\pi_\theta, \pi_{\text{ref}}, r_\psi, V_\phi$) + состояния оптимизатора + KV-кэш генерации *(в GRPO критик $V_\phi$ отсутствует)*.

### Сравнительный анализ версий PPO и предшественников

|      Метод      |                             Механизм сдерживания шага                              |             Опорная политика в KL              | Критик $V_\phi$ | Несколько эпох |
| :-------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------: | :-------------: | :------------: |
|     **A2C**     |                                        Нет                                         |                      Нет                       |       Да        |      Нет       |
|    **TRPO**     | Жёсткое $\overline{\mathrm{KL}}(\pi_{\text{old}} \parallel \pi_\theta) \le \delta$ |          $\pi_{\theta_{\text{old}}}$           |       Да        |      Нет       |
| **PPO-Penalty** |    Штраф $-\beta\,\mathrm{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta)$     | $\pi_{\theta_{\text{old}}}$ *(адапт. $\beta$)* |       Да        |       Да       |
|  **PPO-Clip**   |           Клиппинг $r_t \in [1 - \varepsilon; 1 + \varepsilon]$ и $\min$           |            Нет *(опц. early stop)*             |       Да        |       Да       |
|  **RLHF-PPO**   |     PPO-Clip + токенный штраф $-\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$      |       $\pi_{\text{ref}}$ *(обратный KL)*       |       Да        |       Да       |
|    **GRPO**     | PPO-Clip по токенам + $-\beta\,\mathrm{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ |      $\pi_{\text{ref}}$ *(оценка $k_3$)*       |       Нет       |       Да       |

### +ы & -ы PPO

> [!SUCCESS] Преимущества
> * Только градиенты первого порядка: простая реализация поверх Adam, совместимость с общим backbone, трансформерами и распределённым обучением.
> * Устойчивость к гиперпараметрам *(дефолт $\varepsilon = 0.2$, $\lambda = 0.95$ работает на широком классе сред)*.
> * Переиспользование батча за счёт $K$ эпох *(выше sample efficiency, чем у A2C и TRPO)*.

> [!FAILURE] Недостатки
> * Остаётся on-policy алгоритмом: роллауты выбрасываются после $K$ эпох $\implies$ уступает off-policy методам по sample efficiency.
> * В отличие от TRPO, не даёт строгой гарантии монотонного улучшения: клиппинг гасит градиент отдельных сэмплов, не запрещая политике выйти за границу по совокупности шагов.
> * Качество чувствительно к деталям реализации *(нормировка Advantage, GAE, клиппинг градиентов, выбор числа эпох $K$)*.

---

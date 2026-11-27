При больших или непрерывных пространствах действий $\mathcal{A}$ поиск $\max\limits_{a \in \mathcal{A}} Q(s, a)$ в [[SARSA и Q-learning|Value-based методах]] становится вычислительно невозможным. Методы градиента стратегии напрямую параметризуют стратегию $\pi_\theta(a \mid s)$ и оптимизируют её веса $\theta$ градиентным подъёмом.

==Опр.== **Целевая функция эффективности стратегии** - математическое ожидание суммарной дисконтированной награды при следовании параметрической стратегии $\pi_\theta$:$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum\limits_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right] \to \sup\limits_\theta$$
==Опр.== **Policy Gradient** *(градиент стратегии)* - класс model-free методов RL, в которых стратегия задаётся дифференцируемой параметрической моделью $\pi_\theta(a \mid s)$, а параметры $\theta$ обновляются методом градиентного подъёма для максимизации $J(\theta)$.

### Лемма о нулевом матожидании градиента логарифма стратегии
$$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \right] = 0$$
#### Доказательство

1. По определению математического ожидания и тождеству $\nabla_\theta \log \pi_\theta(a \mid s) = \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)}$:$$\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \right] = \int\limits_{\mathcal{A}} \frac{\nabla_\theta \pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} \pi_\theta(a \mid s) \, da = \int\limits_{\mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) \, da$$
2. Вынося градиент за знак интеграла *(по [[Теорема Лебега о мажорируемой сходимости|Т. Лебега]])*:$$\int\limits_{\mathcal{A}} \nabla_\theta \pi_\theta(a \mid s) \, da = \nabla_\theta \int\limits_{\mathcal{A}} \pi_\theta(a \mid s) \, da = \nabla_\theta 1 = 0$$

### Теорема о градиенте стратегии
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum\limits_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$
#### Доказательство

1. Запишем целевой функционал как интеграл по пространству всех траекторий $\tau$ и внесём оператор градиента $\nabla_\theta$ под знак интеграла (носитель траекторий задан MDP и не зависит от $\theta$, $|R(\tau)| \le \min\left\{T r_{\max},\ \frac{r_{\max}}{1 - \gamma}\right\}$ *(где $r_{\max} = \sup |R|$)*, а $\nabla_\theta p = p \sum\limits_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$; при равномерно ограниченном $\nabla_\theta \log \pi_\theta$ и локально интегрируемом $\sup\limits_{\theta' \in U} p(\tau \mid \theta')$ $\exists$ мажоранта подынтегрального выражения $\implies$ по [[Теорема Лебега о мажорируемой сходимости|Т. Лебега]] перестановка интеграла и градиента возможна):$$\nabla_\theta J(\theta) = \nabla_\theta \int p(\tau \mid \theta) R(\tau) \, d\tau = \int \nabla_\theta p(\tau \mid \theta) R(\tau) \, d\tau$$
2. Применим тождество $\nabla_\theta p(\tau \mid \theta) = p(\tau \mid \theta) \nabla_\theta \log p(\tau \mid \theta)$:$$\nabla_\theta J(\theta) = \int p(\tau \mid \theta) \nabla_\theta \log p(\tau \mid \theta) R(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p(\tau \mid \theta) R(\tau) \right]$$
3. Разложим вероятность траектории $\tau = (s_0, a_0, s_1, \dots, s_T)$ в произведение по марковскому свойству:$$p(\tau \mid \theta) = p(s_0) \prod\limits_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P_{s_t a_t}(s_{t+1})$$
4. Прологарифмируем $p(\tau \mid \theta)$ и возьмём градиент по $\theta$. Т.к. начальное распределение $p(s_0)$ и динамика среды $P_{s_t a_t}(s_{t+1})$ не зависят от $\theta$, их градиенты равны нулю:$$\log p(\tau \mid \theta) = \log p(s_0) + \sum\limits_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum\limits_{t=0}^{T-1} \log P_{s_t a_t}(s_{t+1}) \implies \nabla_\theta \log p(\tau \mid \theta) = \sum\limits_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$
5. Подставим полученную сумму и $R(\tau) = \sum\limits_{k=0}^{T-1} \gamma^k R(s_k, a_k)$ в математическое ожидание:$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum\limits_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) \left( \sum\limits_{k=0}^{T-1} \gamma^k R(s_k, a_k) \right) \right]$$
6. При $k < t$ награда $R(s_k, a_k)$ определяется предысторией $H_t = (s_0, a_0, \dots, s_t)$ и не зависит от $a_t$. По формуле полного математического ожидания и Л. о нулевом матожидании градиента логарифма стратегии: $$\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) R(s_k, a_k)] = \mathbb{E}_{H_t}\left[R(s_k, a_k) \, \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \mid H_t]\right] = 0$$
7. Значит в сумме остаются только слагаемые с $k \ge t$; вынося $\gamma^t$ и вводя возврат от момента $t$ как $G_t = \sum\limits_{k=t}^{T-1} \gamma^{k-t} R(s_k, a_k)$, получаем:$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum\limits_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$
8. Применяя формулу полного математического ожидания при условии $(s_t, a_t)$ и определение Q-функции, получаем искомое равенство:$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum\limits_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$

> [!TIP] Трюк с логарифмом
> Вычислить $\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}R(\tau)$ напрямую нельзя, т.к. распределение траекторий $p(\tau \mid \theta)$ зависит от $\theta$, а вероятности переходов среды $P_{sa}(s')$ неизвестны или недифференцируемы. Тождество $\nabla_\theta p(\tau \mid \theta) = p(\tau \mid \theta) \nabla_\theta \log p(\tau \mid \theta)$ переносит градиент внутрь математического ожидания, превращая его в среднее по траекториям, оцениваемое методом Монте-Карло без знания модели среды.

### Алгоритм REINFORCE

Алгоритм REINFORCE оценивает градиент $\nabla_\theta J(\theta)$ по полным траекториям методом Монте-Карло, подставляя эмпирический возврат $G_t$ вместо $Q^{\pi_\theta}(s_t, a_t)$.

Вход: дифференцируемая стратегия $\pi_\theta(a \mid s)$, темп обучения $\alpha > 0$.
1. Инициализировать параметры стратегии $\theta$ случайно.
2. Повторять до сходимости:
	1. Сгенерировать батч из $N$ траекторий $\tau^{(i)} = (s_0^{(i)}, a_0^{(i)}, r_0^{(i)}, \dots, s_T^{(i)}) \sim \pi_\theta$.
	2. Для каждой траектории $i \in \overline{1,N}$ и шага $t \in \overline{0, T-1}$ вычислить дисконтированный возврат:$$G_t^{(i)} = \sum\limits_{k=t}^{T-1} \gamma^{k-t} r_k^{(i)}$$
	3. Вычислить оценку градиента по батчу:$$\hat{g} = \frac{1}{N} \sum\limits_{i=1}^N \sum\limits_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)}) G_t^{(i)}$$
	4. Выполнить шаг градиентного подъёма:$$\theta \leftarrow \theta + \alpha \hat{g}$$

Геометрически вектор $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ задаёт направление в пространстве параметров $\theta$, быстрее всего увеличивающее логарифм вероятности действия $a_t$ в состоянии $s_t$. Умножение на возврат $G_t$ усиливает вероятности действий, приведших к высокой суммарной награде, и подавляет неудачные траектории.

> [!tip] Эвристика
> Отбрасывание множителя $\gamma^t$ в шаге 2.3 даёт популярную на практике эвристику, которая строго не является градиентом исходной целевой функции.

> [!WARNING] Проблема распределения кредита доверия
> При получении сигнала награды в конце эпизода агенту сложно определить, какое именно из прошлых действий $a_t$ привело к успеху: алгоритм поощряет или штрафует все действия траектории пропорционально общему возврату $G_t$, что порождает высокую дисперсию оценки градиента.

> [!EXAMPLE] Пример: **Дрессировка и отложенный сигнал**
> Обучение методом REINFORCE аналогично дрессировке, где животное выполняет длинную цепочку команд, а лакомство получает только в финале. Поощряются все выполненные действия пропорционально итоговой награде, хотя решающими могли быть лишь отдельные шаги.

#### Асимптотика

Для одной итерации алгоритма REINFORCE по батчу из $N$ траекторий длины $T$:
1. Сложность по времени: $O(N \cdot T \cdot (C_{\text{env}} + C_{\text{forward}} + |\theta|))$, где $C_{\text{env}}$ - время шага среды, $C_{\text{forward}}$ - время прямого прохода сети стратегии, $|\theta|$ - число параметров модели *(вычисление всех возвратов $G_t^{(i)}$ обратным проходом с конца траектории требует $O(N \cdot T)$ операций)*.
2. Сложность по памяти: $O(N \cdot T \cdot (\dim \mathcal{S} + \dim \mathcal{A}) + |\theta|)$ - хранение сгенерированных батчей траекторий и графа вычислений.

### +ы & -ы Policy Gradient (REINFORCE)

> [!SUCCESS] Преимущества
> * Прямая оптимизация целевой функции $J(\theta)$ без необходимости построения модели среды *(model-free)* и без требования дифференцируемости функции награды.
> * Естественная работа с непрерывными пространствами действий $\mathcal{A}$ и стохастическими стратегиями.
> * Гарантии сходимости к локальному оптимуму при корректном выборе шага обучения $\alpha$.

> [!FAILURE] Недостатки
> * Высокая дисперсия оценки градиента, требующая сбора большого числа траекторий $N \cdot T$ и снижающая эффективность использования данных.
> * Строгий on-policy режим: необходимость генерировать полные эпизоды текущей стратегией $\pi_\theta$ и несовместимость с [[SARSA и Q-learning|Experience Replay]] в чистом виде.
> * Чувствительность к проблеме распределения кредита доверия *(требует введения бейзлайнов или перехода к архитектурам Actor-Critic)*.

---

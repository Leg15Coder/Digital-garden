При планировании поведения на основе [[Введение в обучение с подкреплением|функции ценности состояний]] $V^\pi(s)$ для выбора оптимального действия в состоянии $s \in \mathcal{S}$ агенту необходимо вычислять $\sum\limits_{s' \in \mathcal{S}} P_{sa}(s') V^\pi(s')$, что невозможно без знания переходных вероятностей среды $P_{sa}(s')$. Переход к оценке пар «состояние-действие» позволяет выбирать действия напрямую из опыта взаимодействия со средой *(model-free [[Введение в обучение с подкреплением|RL]])*.

==Опр.== **Q-функция** *(функция ценности действия, Action-Value Function, $Q^\pi(s, a)$)* - математическое ожидание суммарной дисконтированной награды при совершении действия $a \in \mathcal{A}$ в состоянии $s \in \mathcal{S}$ и дальнейшем следовании [[Введение в обучение с подкреплением|стратегии]] $\pi$:$$Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum\limits_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s, a_t = a \right]$$
==Опр.== **Оптимальная Q-функция** *($Q^*(s, a)$)* - наибольшая по всем допустимым стратегиям функция ценности пары $(s, a)$, удовлетворяющая [[Value Iteration|уравнению оптимальности Беллмана]]:$$Q^*(s, a) = \max\limits_\pi Q^\pi(s, a) = \mathbb{E}_{s' \sim P_{sa}} \left[ r(s, a) + \gamma \max\limits_{a' \in \mathcal{A}} Q^*(s', a') \right]$$

### Режимы обучения и исследование среды

==Опр.== **On-policy обучение** *(обучение на текущей стратегии)* - режим обучения, при котором поведенческая стратегия $\mu$, генерирующая траектории в среде, совпадает с целевой оптимизируемой стратегией $\pi$ ($\mu \equiv \pi$).

==Опр.== **Off-policy обучение** *(обучение вне текущей стратегии)* - режим обучения, при котором агент оценивает и улучшает целевую стратегию $\pi$, используя данные, собранные другой поведенческой стратегией $\mu$ ($\mu \neq \pi$).

==Опр.== **$\varepsilon$-мягкая стратегия** *($\varepsilon$-soft policy)* - стохастическая стратегия $\pi$, удовлетворяющая условию $\forall s \in \mathcal{S}\ \forall a \in \mathcal{A}\ :\ \pi(a \mid s) \ge \frac{\varepsilon}{|\mathcal{A}|}$.

==Опр.== **$\varepsilon$-жадная стратегия** *($\varepsilon$-greedy policy)* - частный случай $\varepsilon$-мягкой стратегии, выбирающий случайное действие из $Uniform(\mathcal{A})$ с вероятностью $\varepsilon$ и жадное действие $\arg\max\limits_{a \in \mathcal{A}} Q(s, a)$ с вероятностью $1 - \varepsilon$.

### Оптимальная $\varepsilon$-мягкая стратегия

==Утв.== Стратегия $\pi$ является оптимальной в классе $\varepsilon$-мягких стратегий $\iff \forall s \in \mathcal{S}\ \forall a \notin \operatorname{Argmax}\limits_{a' \in \mathcal{A}} Q^\pi(s, a')\ :\ \pi(a \mid s) = \frac{\varepsilon}{|\mathcal{A}|}$.

#### Доказательство

1. Зафиксируем состояние $s \in \mathcal{S}$ и запишем задачу условной линейной оптимизации по распределению $\pi(\cdot \mid s)$ при фиксированной оценке будущего $Q^*_{\varepsilon\text{-soft}}(s, a)$:$$\begin{cases} \sum\limits_{a \in \mathcal{A}} \pi(a \mid s) Q^*_{\varepsilon\text{-soft}}(s, a) \to \max\limits_\pi \\ \sum\limits_{a \in \mathcal{A}} \pi(a \mid s) = 1, \quad \forall a \in \mathcal{A}\ :\ \pi(a \mid s) \ge \frac{\varepsilon}{|\mathcal{A}|} \end{cases}$$
2. Введём нормированную остаточную стратегию $\tilde{\pi}(a \mid s) = \frac{\pi(a \mid s) - \frac{\varepsilon}{|\mathcal{A}|}}{1 - \varepsilon}$. Из условий задачи $\forall a \in \mathcal{A}\ :\ \tilde{\pi}(a \mid s) \ge 0$ и $\sum\limits_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) = \frac{1 - \varepsilon}{1 - \varepsilon} = 1$, т.ч. $\tilde{\pi}$ - корректное вероятностное распределение.
3. Выражая $\pi(a \mid s) = (1 - \varepsilon) \tilde{\pi}(a \mid s) + \frac{\varepsilon}{|\mathcal{A}|}$, приведём целевую функцию к виду:$$(1 - \varepsilon) \sum\limits_{a \in \mathcal{A}} \tilde{\pi}(a \mid s) Q^*_{\varepsilon\text{-soft}}(s, a) + \frac{\varepsilon}{|\mathcal{A}|} \sum\limits_{a \in \mathcal{A}} Q^*_{\varepsilon\text{-soft}}(s, a)$$
4. Максимум первой суммы достигается $\iff$ вся вероятностная масса $\tilde{\pi}(a \mid s)$ сосредоточена на действиях $a^* \in \operatorname{Argmax}\limits_{a' \in \mathcal{A}} Q^*_{\varepsilon\text{-soft}}(s, a')$, откуда $\forall a \notin \operatorname{Argmax}\limits_{a' \in \mathcal{A}} Q^*_{\varepsilon\text{-soft}}(s, a')\ :\ \tilde{\pi}(a \mid s) = 0 \implies \pi(a \mid s) = \frac{\varepsilon}{|\mathcal{A}|}$.

---

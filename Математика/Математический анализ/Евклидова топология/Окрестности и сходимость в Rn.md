Пусть в $\mathbb{R}^d$ задана стандартная [[Евклидова структура|евклидова норма]] $|x|$.

==Опр.== **$\varepsilon$-окрестность** *(открытый шар радиуса $\varepsilon > 0$ с центром в т. $x \in \mathbb{R}^d$)*:$$U_\varepsilon(x) \equiv B_\varepsilon(x) = \{y \in \mathbb{R}^d \mid |y - x| < \varepsilon\}$$

==Опр.== **Проколотая $\varepsilon$-окрестность** т. $x \in \mathbb{R}^d$:$$\mathring{U}_\varepsilon(x) = U_\varepsilon(x) \setminus \{x\}$$

==Опр.== **Дополнение множества** $E \subset \mathbb{R}^d$:$$E^c = \mathbb{R}^d \setminus E$$

==Опр.== Последовательность $\{x_n\} \subset \mathbb{R}^d$ **сходится к** $x \in \mathbb{R}^d$ ($x_n \xrightarrow[n \to \infty]{} x$), если $\lim\limits_{n \to \infty} |x_n - x| = 0$.

==Утв.== Сходимость последовательности в $\mathbb{R}^d$ равносильна покоординатной сходимости:$$\lim\limits_{n \to \infty} x_n = x \iff \forall k \in \overline{1, d}\ :\ x_n(k) \xrightarrow[n \to \infty]{} x(k)$$

#### Доказательство
1. ($\Rightarrow$) $\max\limits_{k \in \overline{1, d}} |x_n(k) - x(k)| \le \sqrt{\sum\limits_{i=1}^d |x_n(i) - x(i)|^2} = |x_n - x| \xrightarrow[n \to \infty]{} 0$.
2. ($\Leftarrow$) $\lim\limits_{n \to \infty} |x_n - x|^2 = \lim\limits_{n \to \infty} \sum\limits_{i=1}^d |x_n(i) - x(i)|^2 = \sum\limits_{i=1}^d \lim\limits_{n \to \infty} |x_n(i) - x(i)|^2 = 0$.

---

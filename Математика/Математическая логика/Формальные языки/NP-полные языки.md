## Теорема о существовании [[Полиномиальные сводимости языков#Трудные и полные языки|NPC]]

- $L_{NP} = \{(M, x, t)\ |\ \text{НМТ}\ M\ \text{принимает}\ x\ \text{за}\ t\ \text{тактов}\} \in NPC$

#### Доказательство

Пусть $A \in NP$, пусть НМТ $M_A$ разрешает $A$ за время $T_A(n) = poly(n)$, тогда пусть $f(x) = (M_A, x, T_A(|x|))$ - данная функция работает за $T_A(|x|) + |M_A| = T_A(|x|) + const = poly(|x|)$ $\Rightarrow$ $(x \in A \Leftrightarrow f(x) \in L_{NP})$ $\Rightarrow$ $A \leq_p L_{NP}$ $\Rightarrow$ $L_{NP} \in NP\text{-}Hard$

$L_{NP}$ можно разрешить на недетерминированном эмуляторе, который работает $O(t^2)$ тактов (по [[Теоремы о времени моделирования на универсальной МТ|теореме о времени моделировании]]) $\Rightarrow$ $L_{NP} \in NP$ $\Rightarrow$ $L_{NP} \in NPC$

## Перечень известных $NPC$ языков

1) $SAT$
	- Это опорный $NPC$ язык, доказательство полноты которого приведено в [[Теорема Кука-Левина|теореме Кука-Левина]]
2) $3\text{-}SAT$
3) $VC$
4) [[INDSET|IS]]
5) $CLIQUE$
6) $3\text{-}COL$
7) $DHAMPATH$
8) $UHAMPATH$
9) $DHAMCYCLE$
10) $UHAMCYCLE$
11) [[TSP]]
12) $EXACTCOVER$
13) $SUBSETSUM$
14) $KNAPSACK$

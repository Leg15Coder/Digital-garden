==Опр.== **AVL-дерево** - [[Дерево поиска|бинарное дерево поиска]], в котором $\forall$ вершины $v\ :\ |h(v_l) - h(v_r)| \leq 1$, где $v_l$ и $v_r$ - левый и правый ребёнок $v$ соотв., $h(x)$ - высота (глубина) поддерева, корень которого в вершине $x$.

## Оценка высоты: высота AVL-дерева на $n$ вершинах равна $O(\log n)$

Пусть $S(h)$ - минимальное число вершин в AVL-дереве высоты $h$, тогда
1) $S(1) = 1$ 
2) $S(2) = 2$
3) $S(h) = 1 + S(h - 1) + S(h - 2)$ ($h-1$ и $h-2$ - минимальные высоты поддеревьев)

Из чего следует, что $S(h) = F_{h+2}-1$, где $F_n$ - $n$-е число фибоначчи $\Rightarrow$ так как $n \geq S(h)$, то $h = O(\log n)$. *(подробные рассуждения см. в фибоначчиевой куче)*

## Балансировка

Балансировка - процесс изменения структуры дерева с целью сохранения его основного свойства. В AVL-дереве балансировка происходит за счёт поворота рёбер, которых есть 4 типа:
- Малый левый поворот
- Малый правый поворот
- Большой левый поворот
- Большой правый поворот

![[Pasted image 20250728174148.png]]

Каждый из малых поворотов будем считать за элементарную операцию в дереве поиска, поскольку выполняются они за $O(1)$. Пусть $\Delta(v) = h(v_l) - h(v_r)$.

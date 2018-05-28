latex代码

oversampling with vae

```
\begin{algorithm}
  \caption{Oversampling using VAE}
  \label{alg:vae}
  \SetAlgoNoLine
  \KwIn{A dataSet $X$,sampling rate $k$}
  \KwOut{$x_{new}$ after oversampling}
  \BlankLine
  $X$ $\leftarrow$ $\frac{X-\overline{X}}{s}$ \\
  $X_{train},X_{test}$ $\leftarrow$ divide($X$) \\
  \For{each feature $j$ in X}{
  $nelements_{j}$ $\leftarrow$ $\sum_1^{N_{+}} distinct{X_{j}}$
  }
  Decide $X_{trainvae}$ with formula (10) \\
  $vae$ $\leftarrow$ trainvae($X_{trainvae}$) \\
  $X_{ov}$ $\leftarrow$ sample($vae$) \\
  Synthesize $X_{final}$ with formula (11) \\
  $X_{new}$ $\leftarrow$ $X_{final}\bigcup X_{train}$ \\
  return $X_{new}$
\end{algorithm}

```

computing the igir

```
\begin{algorithm}
  \caption{Computing the IGIR}
  \label{alg:IGIR}
  \SetAlgoNoLine
  \KwIn{A dataSet $X$,label $Y$,number of nearest neighbors $k$ in $k$-NN}
  \KwOut{IGIR}
  \BlankLine
  \For{$x$ in $X$ with label $y_{x}$}{
    {$M$ $\leftarrow$ the $k$ nearest neighbors of $x$} \\
    $t_{k}(x)$ $\leftarrow$ $\frac{1}{k}$ $\sum{weight*IR(x,M)}$
  }
 $wei-T_{-}$ $\leftarrow$ $\frac{1}{N_{-}}$ $\sum t_{k}(x)*sgn(y_{x}==0)$ \\
 $wei-T_{+}$ $\leftarrow$ $\frac{1}{N_{+}}$ $\sum t_{k}(x)*sgn(y_{x}==1)$ \\
 IGIR $\leftarrow$ $\sqrt{wei-T_{-}*wei-T_{+}}$ \\
 return IGIR
\end{algorithm}
```


function NewSolution = LocalSearch(solution, d, r, NVAR,Correlations)

    Selected = OneCounter(solution,NVAR);

    if(Selected == d)
        NewSolution = Ripple_Add(r, d, solution, Correlations);
        NewSolution = Ripple_Remove(r, d, NewSolution, Correlations);
    end

    if(Selected > d)
        NewSolution = Ripple_Remove(r, d, solution, Correlations);
    end

    if(Selected < d)
        NewSolution = Ripple_Add(r, d, solution, Correlations);
    end
end
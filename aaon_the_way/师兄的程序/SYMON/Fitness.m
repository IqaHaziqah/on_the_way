    function AUC = Fitness(sol,NVAR,Train,Test,tr_lbl,ts_lbl,trSize,tsSize)
    
        position = PositionDetector(sol,NVAR);
        Ones = OneCounter(sol,NVAR);
        [GM, CA, AUC, Fmeasure] = mySVM(Train,Test,tr_lbl,ts_lbl,position,Ones,trSize,tsSize);
        
    end
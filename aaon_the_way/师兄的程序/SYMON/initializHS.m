function [HM,F] = initialize_HS(HMS,NVAR,Train,Test,tr_lbl,ts_lbl,trSize,tsSize)
        for i=1:HMS
            for j=1:NVAR
                if(rand > 0.5)
                    HM(i,j)= 1;
                else
                    HM(i,j)= 0;
                end
            end
            if(OneCounter(HM(i,:),NVAR)>0)
                F(i) = Fitness(HM(i,:),NVAR,Train,Test,tr_lbl,ts_lbl,trSize,tsSize);
            else
                F(i) = 0;
            end
        end
    end
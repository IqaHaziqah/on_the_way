function [HM, F] = UpdateHM( NewFit,fitness,HM,HMS,NCHV,NVAR )

            WorstIndex = 1;
            WorstFit=fitness(1);
            for i = 2:HMS
                if( fitness(i) < WorstFit )
                    WorstFit = fitness(i);
                    WorstIndex = i;
                end
            end
            
            if( NewFit > WorstFit )
                fitness(WorstIndex)=NewFit;
                for i = 1:NVAR
                    HM(WorstIndex,i)=NCHV(i);
                end
            end
            F = fitness;
end
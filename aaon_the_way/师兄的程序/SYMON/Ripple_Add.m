function NewSolution = Ripple_Add(r, d, NHV, Correlations)
    SelectedCores=[];
    Len = OneCounter(NHV,length(NHV));
    while(d ~= Len)
                
        Zeropos = ZeroPositionDetector(NHV,length(NHV));
        Onepos = PositionDetector(NHV,length(NHV));
        
        for i=1:length(Onepos)
            SelectedCores(i) = Correlations(Onepos(i));
        end
        
        for i=1:length(Zeropos)
            IgnoredCores(i) = Correlations(Zeropos(i));
        end
        
        CorrelationSelected = sort(SelectedCores,'ascend');
        CorrelationIgnored = sort(IgnoredCores,'descend');
        
        for R = 1:r
            slct = CorrelationIgnored(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == slct)
                    NHV(i) = 1;
                    Len = Len + 1;
                    break;
                end
            end
        end
        
         for R = 1:(r-1)
            rmv = CorrelationSelected(R);
            for i= 1:length(Correlations)
                if(Correlations(i) == rmv)
                    NHV(i) = 0;
                    Len = Len - 1;
                    break;
                end
            end
         end
    end
    NewSolution = NHV;
end
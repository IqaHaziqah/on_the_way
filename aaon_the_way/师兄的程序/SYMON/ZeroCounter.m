function Zeros = ZeroCounter(NHV,NVAR)
Zeros = 0;

  for l = 1: NVAR
    if (NHV(l) == 0)
      Zeros = Zeros + 1;
    end
  end
end
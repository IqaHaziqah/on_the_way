function Ones = OneCounter(NHV,NVAR)
Ones = 0;

  for l = 1: NVAR
    if (NHV(l) == 1)
      Ones = Ones + 1;
    end
  end
end
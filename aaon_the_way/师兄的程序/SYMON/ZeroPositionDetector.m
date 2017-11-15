function position = ZeroPositionDetector(NHV,NVAR)
p = 1;

  for l = 1: NVAR
    if (NHV(l) == 0)
      position(p) = l;
      p = p + 1;
    end
  end
end
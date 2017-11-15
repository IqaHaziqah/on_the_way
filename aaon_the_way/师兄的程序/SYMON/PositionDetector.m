function position = PositionDetector(NHV,NVAR)
p = 1;
y=[];
  for l = 1: NVAR
    if (NHV(l) == 1)
      y(p) = l;
      p = p + 1;
    end
  end
  position=y;
end
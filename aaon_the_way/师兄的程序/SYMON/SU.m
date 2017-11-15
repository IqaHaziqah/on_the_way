function [Correlations] = SU(data,class_id)
y = class_id;

for x = 1:(class_id - 1)
        S = 2*(entropy(data(:,x)) - conditionalEntropy(data(:,x),data(:,y)));
        M = entropy(data(:,x)) + entropy(data(:,y));
        Correlations(x, 1) = S / M;
        Correlations(x, 2) = x;
end
    Correlations(x, 1) = Correlations(x, 1) / sum(Correlations(x, 1));
end

function val=StopCondition(Itr)
    global MaxItr;
    val = 1;
    if(Itr>MaxItr)
        val=0;
    end
end
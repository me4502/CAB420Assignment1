    % Constructor (takes zero arguments or 3)
    function obj = knnRegress(K,Xtr,Ytr)
      obj.K=1; obj.Xtrain=[]; obj.Ytrain=[];
      obj=class(obj,'knnRegress');
      if (nargin > 0) 
        obj.K = K;
        obj.Xtrain = Xtr;
        obj.Ytrain = Ytr;
      end;
    end


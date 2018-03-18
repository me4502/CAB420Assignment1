    % Constructor (takes no arguments or training data)
    function obj = linearReg(Xtr,Ytr, varargin)
      obj.theta=[];
      obj=class(obj,'linearReg');
      if (nargin > 0) 
        obj=train(obj,Xtr,Ytr, varargin{:});
      end;
    end


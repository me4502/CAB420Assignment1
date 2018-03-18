    % calculate mean absolute error for a given validation data set
    function err = mae(obj,Xval,Yval)
      Yhat = obj.predict(Xval);
      err = mean( abs(Yhat-Yval) );
    end

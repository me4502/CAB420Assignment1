function Xext = polyx(X,P)
% create polynomial features of a single (univariate) feature

[m,n]=size(X);
if (n~=1) error('PolyX currently only works for univariate data'); end;
Xext = zeros(m,P);
for p=0:P,
  Xext(:,p+1)=X.^p;
end;


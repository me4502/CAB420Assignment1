function class2DPlot(learner,X,Y)

figure; plot(X(:,1),X(:,2),'k.');
ax=axis;  % get current plot appearance
N=256;    % density of evaluation

% Evaluate each point of feature space and predict the class
X1 = linspace(ax(1),ax(2),N); X1sp=X1'*ones(1,N);
X2 = linspace(ax(3),ax(4),N); X2sp=ones(N,1)*X2;
pred = predict(learner,[X1sp(:),X2sp(:)]);

% plot decision values for the space in "faded" color
cmap=jet(256); 
figure; imagesc(X1,X2,reshape(pred,N,N)'); axis xy; hold on; colormap(cmap*.4+.6);

% plot each classes' data in the full spectrum color
for c=unique(Y)', 
  col= fix((c-min(pred))/(max(pred)-min(pred))*255+1);
  %H=plot(X(find(Y==c),1),X(find(Y==c),2),'ko','markersize',7,'markerfacecolor',cmap(col,:)); 
  H=plot(X(find(Y==c),1),X(find(Y==c),2),'o','markersize',7,'color',cmap(col,:),'markerfacecolor',cmap(col,:)); 
  %set(H,'markerfacecolor',cmap(col,:));
end;


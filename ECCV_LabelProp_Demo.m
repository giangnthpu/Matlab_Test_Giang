function [Accuracy,Retrieval]=ECCV_LabelProp_Demo;
%
%
WINDOWS=300;ooooooooooooooooooooooooooooooooooooooooooooooooooooo
load Label_propagation_Mpeg.mat;
NumOfShapes=1400;% 
Num_retrieval=40;
NO_ShapeClass=20;
Diff=(Diff+Diff')/2;
[T,INDEX]=sort(Diff,2);
[m,n]=size(Diff);
Retrieval=zeros(NumOfShapes,Num_retrieval);
correct=0;
K=0.27;
Neighbor=10;
for item=1:NumOfShapes
    item
    TEMP=INDEX(item,1:WINDOWS);% choose the most WINDOWS similar shapes for the query shape
    Real_Diff=zeros(WINDOWS,WINDOWS);
    Real_Diff=Diff(TEMP,TEMP);% Build up the new matrix for the most WINDOWS similar shapes
    TT=sort(Real_Diff,2);
    for k=1:WINDOWS% calculate sigma for affinity matrix
        for j=1:WINDOWS
            SIGMA=mean([TT(k,2:Neighbor),TT(j,2:Neighbor)]);
            W(k,j)=normpdf(Real_Diff(k,j),0,K*SIGMA);
        end
    end
    [m,n]=size(W);
    Norm=repmat(sum(W')',1,n);
    P=W./Norm;% normalization
    f=zeros(WINDOWS,1);
    %------------------ Learning the distance-------------------------
    f(1)=1;
    for k=1:10
        f=P*f;
        f(1)=1;
    end
    [TT,TEMP_R]=sort(f,'descend');
    Results=TEMP(TEMP_R);
    %------------------Bulleye score for MPEG7, could be modified for other datasets--------------------------------------------------
    Retrieval(item,:)=Results(1:Num_retrieval);
    for t=1:40
        if ceil(item/NO_ShapeClass)==ceil(Results(t)/NO_ShapeClass)
            correct=correct+1;
        end
    end
end
Accuracy=correct/(NO_ShapeClass*NumOfShapes);
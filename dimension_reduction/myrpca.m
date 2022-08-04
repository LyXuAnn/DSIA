load('/home/yikui/DR_DATA/sample2000.mat');
sample_num = 2000;
sample_dimension = 54776;

for i=1:sample_num
    sample(:,i)=sample(:,i)/norm(sample(:,i));
end

%mean the sample data
mean_vec=repmat(sum(sample,2)/sample_num,1,sample_num);
sample=sample-mean_vec;
sample = sample';

%noise
%spr=0.0005;%test 3
%pp=randperm(sample_dimension*sample_num);
% p=randperm(2400);
%LL = round(spr*sample_dimension*sample_num);
% L=round(spr*2400);
%maxB=max(abs(sample(:)));
%noise=zeros(sample_num,sample_dimension);
%noise(pp(1:LL))=randn(LL,1);
% index=p(1:L);
%if noise_type
%    noise(pp(1:LL))=maxB*sign(randn(LL,1)); 
%end
%sample=sample+noise;%test 3

% %
tic;
[A, E] = exact_alm_rpca(sample',0.005);
toc;

[U S V] = svd(A);

lamdasrpca2000 = diag(S);

U70 = U(:,1:100);
%dlmwrite('U50_73_50_0.001.txt',U50'','delimiter', ',','precision', 12,'newline', 'pc')
%dlmwrite('U_eROMS_RPCA100_2000_0.005.txt',U70,'delimiter', ',','precision', 12,'newline', 'pc')



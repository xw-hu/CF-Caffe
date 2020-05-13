clear;
clc;

% root = '/home/xwhu/dataset/ECSSD/images_ECSSD/';
% res = dir(fullfile(root,'*.jpg'));

% root = '/home/xwhu/dataset/HKU-IS/imgs/';
% res = dir(fullfile(root,'*.png'));
% 
% root = '/home/xwhu/dataset/PASCAL-S/input/';
% res = dir(fullfile(root,'*.jpg'));

% root = '/home/xwhu/dataset/THUR15000/image/';
% res = dir(fullfile(root,'*.jpg'));

% gt_root = '/home/xwhu/dataset/THUR15000/image/';
% gt = dir(fullfile(gt_root,'*.png'));

root = '/home/xwhu/dataset/faceshadow/RealTestset/cropped/';
res = dir(fullfile(root,'*.jpg'));

res = res(randperm(length(res)));

% root2 = '/home/xwhu/dataset/soc/Val/';
% res2 = dir(fullfile(root2,'*.jpg'));


fid = fopen('./faceshadow/test_real.txt','w');
%fid2 = fopen('./faceshadow/train_shadow_free.txt','w');

for i=1:length(res)
    
    i
    % fprintf(fid,'Val/%s', res2(i).name);
    %fprintf(fid,' gt/%s', gt(i).name);
    fprintf(fid,'%s', res(i).name);
    fprintf(fid,'\n');
    
%     new_string = strsplit(res(i).name, '-');
%     
%     fprintf(fid2,'%s.jpg',  new_string{1});
%     fprintf(fid2,'\n');
    
end
% 
% for i=1:length(res)
%     
%     i
%     fprintf(fid,'Train/%s', res(i).name);
%     %fprintf(fid,' gt/%s', gt(i).name);
%     fprintf(fid,'\n');
%     
% end

fclose(fid);
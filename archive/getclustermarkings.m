% this function comes from Joe
% https://github.com/jmstujenske/PL_drives_nonaversion_via_SOM_IN/tree/main/kilosort2_utilities

function [cids,cgs]=getclustermarkings(folder)
endswithslash=strcmp(folder(end),'\');
% if ~endswithslash
%     folder=[folder,'\'];
% end
original=fullfile(folder,'cluster_KSlabel.tsv');
[cids, cgs] = readClusterGroupsCSV(original);
new=fullfile(folder,'cluster_group.tsv');
if exist(new,'file')
[cids_new, cgs_new] = readClusterGroupsCSV(new);
cids_all=sort(unique([cids cids_new]));
cgs_all=nan(1,length(cids_all));
% changed=ismember(cids,cids_new);
% changed2=ismember(cids_new,cids);
% cgs(changed)=cgs_new(changed2);
% cids(changed)=cids_new(changed2);

cgs_all(ismember(cids_all,cids))=cgs;
cgs_all(ismember(cids_all,cids_new))=cgs_new;
cids=cids_all;
cgs=cgs_all;
end


% %%
% mask=NaN*ones(size(voxels,2),size(voxels,3));
% lon=ncread('extreme_dy_by_yr_defaults.nc','lon');
% lat=ncread('extreme_dy_by_yr_defaults.nc','lat');
% kk=lon; ll=lat;
% 
% ij=find(ll(:,:)< -60 );
% ji=find(kk(:,:)>0 & kk(:,:)<360); % | kk(:,:)>306 & kk(:,:)<360);
% mask(ji,ij)=1;
% lon=repmat(lon, [1 720]);
% lat=repmat(lat, [1 1440])';
% figure;pcolor(lon,lat,mask);shading flat;fillmap;
load('land_mask.mat'); mask_global=permute(repmat(mask_global,[1 1 38]),[3 1 2]);
vox_detrend=ncread('extreme_dy_by_yr_2019_local_detrend.nc','ndays');
voxels=double(vox_detrend);
voxels=voxels.*mask_global;

%% NEP

mask=NaN*ones(1440,720);
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
NEP_vox=voxels(:,721:1128,361:624);
load('NEP_mask.mat'); NEP_mask=repmat(NEP_mask,[1 1 38]); NEP_mask=permute(NEP_mask,[3 1 2]);
NEP_vox=NEP_vox.*NEP_mask;
NEP_lon=lon(721:1128,361:624);
NEP_lat=lat(721:1128,361:624);
NEP_vox=permute(NEP_vox,[2 3 1]);
%figure;pcolor(NEP_lon,NEP_lat,squeeze(NEP_vox(:,:,1)));shading flat;

vox_mean_det(:,1)=squeeze(nanmean(nanmean(NEP_vox)));
NEP_det=vox_mean_det(:,1);
clearvars -except vox_mean_det voxels NEP_det

%% NWP  ij=find(ll(:,:)<66 & ll(:,:)>0 ); ji=find(kk(:,:)>100 & kk(:,:)<180);
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';

NWP_vox=voxels(:,401:720,361:624);
NWP_lon=lon(401:720,361:624);
NWP_lat=lat(401:720,361:624);
%figure;pcolor(NWP_lon,NWP_lat,squeeze(NWP_vox(30,:,:)));shading flat;fillmap;
NWP_vox=permute(NWP_vox,[2 3 1]);
vox_mean_det(:,2)=squeeze(nanmean(nanmean(NWP_vox)));
NWP_det=vox_mean_det(:,2);
hold on 
clearvars -except vox_mean_det voxels NEP_det NWP_det

%% SP 
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
SP_vox=voxels(:,721:1172,121:360);
SP_lon=lon(721:1172,121:360);
SP_lat=lat(721:1172,121:360);
% figure;pcolor(SP_lon,SP_lat,squeeze(SP_vox(30,:,:)));shading flat;fillmap;

SP_vox=permute(SP_vox,[2 3 1]);
vox_mean_det(:,3)=squeeze(nanmean(nanmean(SP_vox)));
SP_det=vox_mean_det(:,3);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det

%% AUS 
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
AUS_vox=voxels(:,401:720,121:360);
AUS_lon=lon(401:720,121:360);
AUS_lat=lat(401:720,121:360);
%figure;pcolor(AUS_lon,AUS_lat,squeeze(AUS_vox(30,:,:)));shading flat;fillmap;
AUS_vox=permute(AUS_vox,[2 3 1]);
vox_mean_det(:,4)=squeeze(nanmean(nanmean(AUS_vox)));
AUS_det=vox_mean_det(:,4);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det

%% INDIAN
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
IND_vox=voxels(:,81:400,121:480);
IND_lon=lon(81:400,121:480);
IND_lat=lat(81:400,121:480);
%figure;pcolor(IND_lon,IND_lat,squeeze(IND_vox(30,:,:)));shading flat;fillmap;
IND_vox=permute(IND_vox,[2 3 1]);
vox_mean_det(:,5)=squeeze(nanmean(nanmean(IND_vox)));
IND_det=vox_mean_det(:,5);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det

%% SWA 
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
SWA_vox=voxels(:,1161:1368,121:360);
SWA_lon=lon(1161:1368,121:360);
SWA_lat=lat(1161:1368,121:360);
%figure;pcolor(SWA_lon,SWA_lat,squeeze(SWA_vox(30,:,:)));shading flat;fillmap;
SWA_vox=permute(SWA_vox,[2 3 1]);
vox_mean_det(:,6)=squeeze(nanmean(nanmean(SWA_vox)));
SWA_det=vox_mean_det(:,6);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det

%% SEA ij=find(ll(:,:)<0 & ll(:,:)>-60 ); ji=find(kk(:,:)>342 & kk(:,:)<360 | kk(:,:)>0 & kk(:,:)<20);
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
SEA_vox1=voxels(:,1369:1440,121:361);
SEA_lon1=lon(1369:1440,121:361);
SEA_lat1=lat(1369:1440,121:361);
SEA_vox2=voxels(:,1:80,121:361);
SEA_lon2=lon(1:80,121:361);
SEA_lat2=lat(1:80,121:361);

SEA_vox1=permute(SEA_vox1,[2 3 1]);
SEA_vox_mean1=squeeze(nanmean(nanmean(SEA_vox1)));

SEA_vox2=permute(SEA_vox2,[2 3 1]);
SEA_vox_mean2=squeeze(nanmean(nanmean(SEA_vox2)));

% figure;pcolor(SEA_lon1,SEA_lat1,squeeze(SEA_vox1(:,:,30)));shading flat;
% hold on
% pcolor(SEA_lon2,SEA_lat2,squeeze(SEA_vox2(:,:,30)));shading flat;fillmap;
% hold off


vox_mean_det (:,7) = (SEA_vox_mean1 +SEA_vox_mean2)/2;
SEA_det=vox_mean_det (:,7);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det SEA_det

%% NEA (2 areas)
% close all
load('NEA_mask.mat');mask_NEA=permute(repmat(mask_NEA,[1 1 38]),[3 1 2]);
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
NEA_vox1=voxels(:,1:164,361:632);NEA_vox1=mask_NEA.*NEA_vox1;
NEA_lon1=lon(1:164,361:632);
NEA_lat1=lat(1:164,361:632);
NEA_vox2=voxels(:,1281:1440,361:632);
NEA_lon2=lon(1281:1440,361:632);
NEA_lat2=lat(1281:1440,361:632);

NEA_vox1=permute(NEA_vox1,[2 3 1]);
NEA_vox_mean1=squeeze(nanmean(nanmean(NEA_vox1)));

NEA_vox2=permute(NEA_vox2,[2 3 1]);
NEA_vox_mean2=squeeze(nanmean(nanmean(NEA_vox2)));
% figure;pcolor(NEA_lon1,NEA_lat1,squeeze(NEA_vox1(:,:,30)));shading flat;fillmap;
% hold on
% figure;pcolor(NEA_lon2,NEA_lat2,squeeze(NEA_vox2(:,:,30)));shading flat;fillmap;
% hold off

vox_mean_det (:,8)= (NEA_vox_mean1 +NEA_vox_mean2)/2;

NEA_det=vox_mean_det (:,8);

%clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det SEA_det NEA_det

%% NWA 
%close all
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
NWA_vox=voxels(:,1049:1280,361:625);
load('NWA_mask.mat'); NWA_mask=repmat(NWA_mask,[1 1 38]); NWA_mask=permute(NWA_mask,[3 1 2]);
NWA_vox=NWA_vox.*NWA_mask;
NWA_lon=lon(1049:1280,361:625);
NWA_lat=lat(1049:1280,361:625);
 %figure;pcolor(NWA_lon,NWA_lat,squeeze(NWA_vox(38,:,:)));shading flat;
NWA_vox=permute(NWA_vox,[2 3 1]);
vox_mean_det(:,9)=squeeze(nanmean(nanmean(NWA_vox)));
NWA_det=vox_mean_det(:,9);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det SEA_det NEA_det NWA_det

%% ARCTIC (2 areas) ij=find(ll(:,:)>66 ); ji=find(kk(:,:)>0 & kk(:,:)<262 | kk(:,:)>306 & kk(:,:)<360);
mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
ARC_vox=voxels(:,:,625:720);
ARC_lon=lon(:,625:720);
ARC_lat=lat(:,625:720);
ARC_vox=permute(ARC_vox,[2 3 1]);
vox_mean_det(:,10)=squeeze(nanmean(nanmean(ARC_vox)));
%figure;pcolor(ARC_lon,ARC_lat,squeeze(ARC_vox(:,:,30)));shading flat;fillmap;

% % ARC_vox1=voxels(:,1:1048,625:720);
% % ARC_lon1=lon(1:1048,625:720);
% % ARC_lat1=lat(1:1048,625:720);
% % ARC_vox2=voxels(:,1225:1440,625:720);
% % ARC_lon2=lon(1225:1440,625:720);
% % ARC_lat2=lat(1225:1440,625:720);
% % ARC_vox1=permute(ARC_vox1,[2 3 1]);
% % ARC_vox_mean1=squeeze(nanmean(nanmean(ARC_vox1)));
% % ARC_vox2=permute(ARC_vox2,[2 3 1]);
% % ARC_vox_mean2=squeeze(nanmean(nanmean(ARC_vox2)));
% % vox_mean(:,10)= (ARC_vox_mean1 +ARC_vox_mean2)/2;

ARC_det=vox_mean_det(:,10);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det SEA_det NEA_det NWA_det ARC_det

%% ACC 

mask=NaN*ones(size(voxels,2),size(voxels,3));
lon=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lon');
lat=ncread('extreme_dy_by_yr_2019_local_detrend.nc','lat');
lon=repmat(lon, [1 720]);
lat=repmat(lat, [1 1440])';
ACC_vox=voxels(:,:,1:120);
ACC_lon=lon(:,1:120);
ACC_lat=lat(:,1:120);
% figure;pcolor(ACC_lon,ACC_lat,squeeze(ACC_vox(30,:,:)));shading flat;fillmap;
ACC_vox=permute(ACC_vox,[2 3 1]);
vox_mean_det(:,11)=squeeze(nanmean(nanmean(ACC_vox)));
ACC_det=vox_mean_det(:,11);
clearvars -except vox_mean_det voxels NEP_det NWP_det SP_det AUS_det IND_det SWA_det SEA_det NEA_det NWA_det ARC_det ACC_det

close all
function h = plotLabeledPointcloud(pts,labels,fignum)
% This function plots colored 3D points in a figure. The colors are based
% off the associated labels passed in with the pts. The user can pass in a
% figure number to plot on an existing figure. If a figure is not passed in
% a new one will be created and plotted on
% INPUTS:
% pts: an n x 3 matrix of points in [x y z] format
% labels: n x 1 matrix of labels; options are 1004:ground 1100:wire
%                                 1103:pole 1200:ground  1400:facade
% fignum: (optional) figure handle of figure to plot on
%
% OUTPUTS:
% h: handle to figure that the function plotted on
%
% Author: jrb     Date: 11/11/14

if ~exist('fignum', 'var') 
    fignum=figure;
end   

% label colors
veg_col = [0 1 0]; %green
wire_col = [1 0 0]; %red
pole_col = [1 1 0]; %yellow
ground_col = [0 0 1]; %blue
facade_col = [1 1 1]; %white

veg = pts(labels==1004,:); %pull out vegetation pts
wire = pts(labels==1100,:); %pull out wire pts
pole = pts(labels==1103,:); %pull out pole pts
ground = pts(labels==1200,:); %pull out ground pts
facade = pts(labels==1400,:); %pull out facade pts

cld = [veg;wire;pole;ground;facade]; %concatenate pts for plotting
colors = [repmat(veg_col,size(veg,1),1);... %concatenate colors for plotting
               repmat(wire_col,size(wire,1),1);...
               repmat(pole_col,size(pole,1),1);...
               repmat(ground_col,size(ground,1),1);...
               repmat(facade_col,size(facade,1),1);];

% Plot pts
plot_cloud(cld,colors,fignum);

%make background and axis black
set(gcf, 'color', [0 0 0]);
axis equal;
axis off;
set(gca, 'color', [0 0 0]);

end


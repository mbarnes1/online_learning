% handle = plot_cloud(cloud, color, fignum)
%
% Plots a point cloud as a cartesian collection of dots.   
%
% cloud is a [nx3] matrix of (x,y,z) points
%
% color is [nx3] color triples of type single in the range [0,1] for
% per-vertex color or a single [1x3] triple to apply the same color for the
% entire cloud. 
%
% fignum is a handle to an existing figure to plot the cloud data to. Leave
% blank to plot in a new figure window. 
%
% @author uyw
function h = plot_cloud(cloud, color, fignum)
dim = size(cloud);
if dim(1) == 3
    cloud = cloud';
end

if ~exist('fignum', 'var') 
    figure;
    aequal = true;
else
    figure(fignum);
    aequal = false;
end   

hold on

if exist('color', 'var') &&  numel(color) == numel(cloud)   %color exists for each point
%     scatter3(cloud(:,1),cloud(:,2), cloud(:,3), 0.5, ...
%        color / max(color(:)), 'Marker', '.');
   
    %figure;
   
%     p=patch([cloud(:,1)],[cloud(:,2)],[cloud(:,3)], reshape(color, [1 size(color)]), ...
%         'Marker', '.', 'Markersize', 0.5, ...
%         'FaceColor', 'none');

    cloud(isnan(cloud)) = 0;
    num_idx = floor(dim(1) / 3);
    faces = reshape(1:3*num_idx, [3 num_idx]);
    
    h = patch('Vertices',cloud, 'Faces', faces, 'FaceVertexCData',color,...
      'FaceColor','none','EdgeColor','flat',...
      'LineStyle', 'none', ...
      'Marker','.','MarkerFaceColor','flat', 'Markersize', 3);

    
    %'LineStyle', 'none', ...
    % nans added so face will not close
    set(gcf, 'Renderer', 'opengl');
%     set(gcf, 'Renderer', 'zbuffer');
    opengl hardware; 

elseif exist('color', 'var') && numel(color) == 3           %same color for whole cloud
    h = plot3(cloud(:,1),cloud(:,2), cloud(:,3), '.', ...
        'markersize', 0.5, 'Color', color);
else                                                        %use default blue color
    h = plot3(cloud(:,1),cloud(:,2), cloud(:,3), '.', 'markersize', 0.5);
end

xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis');

hold off;

if aequal == true
    axis equal;
    box on;
end
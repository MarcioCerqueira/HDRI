function extOut=fileExtension(filename)
%
%
%        extOut=fileExtension(filename)
%
%
%        Description: return the extension of a file
%
%        Input:
%           -filename: the name of the file
%
%        Output:
%           -extOut: the extension of the file
%
%     Copyright (C) 2011  Francesco Banterle
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%

k = strfind(filename,'.');

%there is no extension
if(size(k)==0)
    error('No file extension');
end

%get the real extension
k = k(end);

extOut = filename((k+1):max(size(filename)));

end

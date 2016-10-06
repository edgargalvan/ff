clear; close all; clc
% Naive line-up optimizer. Just picks best line up according to expert projections

% these are the players that we want to remove
O=[];

% Some Parameters
cmax = 60000;      % Maximum we are allowed to spend
nmax = 9;          % Maximum number of players on roster
n = [1 3 2 1 1 1]; % Maximum number of players per position [QB WR RB TE K DEF]
tmax = 4;          % Maximum number of players that can be on the roster from a single team

% Define decision variable
x = binvar(1,N);

% Number of players not team cost may exceed limit 
Constraints = [sum(x) <= nmax, x*C'<=cmax];

% Limitations on combinations of player positions
for i = 1:6
    Constraints = [Constraints, sum(x(P{i}))<=n(i)]; %#ok<AGROW>
end

% Limitation on fthe number of players from a single team
for i = 1:K
    Constraints = [ Constraints, sum(x(T{i}))<=tmax]; %#ok<AGROW>
end

% Don't put unavailable players on the roster
% Constraints = [ Constraints, sum(x(O))==0];

Objective = -x*A';
options = sdpsettings('verbose',1,'solver','cplex');

%%
nStratagies = 1;
plays = repmat(struct('names',[],'position',[],'fpts',[],'tfpts',[]),nStratagies,1);
for k = 1:nStratagies
    
    sol = solvesdp(Constraints,Objective,options);
    if sol.problem == 0
     solution = double(x);
     solution = logical(solution);
    else
     display('Hmm, something went wrong!');
     sol.info
     yalmiperror(sol.problem)
    end

    % Check Constraints
    c = [sum(solution) <= nmax, solution*C'<=cmax];
    % Limitations on combinations of player positions
    for i = 1:6
        c = [c, sum(solution(P{i}))<=n(i)]; %#ok<AGROW>
    end

    % Limitation on fthe number of players from a single team
    for i = 1:K
        c = [c, sum(solution(T{i}))<=tmax]; %#ok<AGROW>
    end
    
    c = [c sum(solution(O))==0];
    if any(~c)
        error('violating constraint');
    end

    %%
    scores = A(solution);
    plot(find(solution),scores,'or','linewidth',3,'markersize',15);

    fprintf('\r\rObj Val %2.2f, Cost %i\r',solution*A',solution*C')
    fprintf('Who to Play, rank %i:\r\r ',k) 
    winners = [playerList(find(solution)) raw(find(solution'),3)]; %#ok<FNDSB>
    for i = 1:size(winners,1)
        fprintf([winners{i,2} '\t'])
        s = [' ' num2str(scores(i)) '        '];
        fprintf([s(1:5) '\t'])
        fprintf([winners{i,1} '\r'])
    end

    plays(k).names = winners(:,1);
    plays(k).position = winners(:,2);
    plays(k).fpts = scores;
    plays(k).tfpts = sum(scores);
    
    
    Constraints = [ Constraints, sum(x(solution))~=nnz(solution)]; %#ok<AGROW>
end

% save('stratagy','plays')

% Clear the environment
clear; clc; close all;

% Parameters
benefit_per_city = 150; % Benefit for each city visited
cost_factor = 0.2; % Cost factor for the cost calculation

% Read the distance matrix from a CSV file
distance_matrix = csvread('sir_datasets.csv');
num_cities = size(distance_matrix, 1);
dataset_number = 1;

% Initialize benefits and cost matrix
benefits = repmat(benefit_per_city, num_cities, 1);
cost_matrix = distance_matrix * cost_factor;

% Initialize all pairs of starting cities
all_pairs = [];
for i = 1:num_cities
    for j = 1:num_cities
        if i ~= j
            all_pairs = [all_pairs; i, j];
        end
    end
end

% Initialize an empty table for results
results = [];

% Iterate through each starting pair
for pair_index = 1:size(all_pairs, 1)
    starting_cities = all_pairs(pair_index, :);
    
    % Optimize the division of this tour between the two agents using Differential Evolution
    [agent1_path, agent2_path, agent1_benefit, agent2_benefit, agent1_cost, agent2_cost, agent1_timestamps, agent2_timestamps] = optimize_agents_paths_DE(benefits, cost_matrix, distance_matrix, starting_cities);

    % Calculate payoffs
    agent1_payoff = agent1_benefit - agent1_cost;
    agent2_payoff = agent2_benefit - agent2_cost;
    total_payoff = agent1_payoff + agent2_payoff;
    total_benefit = agent1_benefit + agent2_benefit;
    average_payoff = total_payoff / 2;

    % Count the number of cities visited by each agent
    agent1_city_count = numel(agent1_path);
    agent2_city_count = numel(agent2_path);

    % Convert agent paths to strings
    agent1_path_str = num2str(agent1_path, '%d,');
    agent1_path_str = agent1_path_str(1:end - 1); % Remove last comma
    agent2_path_str = num2str(agent2_path, '%d,');
    agent2_path_str = agent2_path_str(1:end - 1); % Remove last comma

    % Append results
    results = [results; {dataset_number, pair_index, starting_cities(1), starting_cities(2), agent1_path_str, agent2_path_str, agent1_benefit, agent2_benefit, agent1_cost, agent2_cost, agent1_payoff, agent2_payoff, total_benefit, total_payoff, average_payoff, agent1_city_count, agent2_city_count, join(string(agent1_timestamps), ','), join(string(agent2_timestamps), ',')}];
end

% Convert to a table for easy saving
resultsTable = cell2table(results, 'VariableNames', {'Dataset_No', 'Pair_Index', 'Starting_City_Agent1', 'Starting_City_Agent2', 'Agent1_Path', 'Agent2_Path', 'Agent1_Benefit', 'Agent2_Benefit', 'Agent1_Cost', 'Agent2_Cost', 'Agent1_Payoff', 'Agent2_Payoff', 'Total_Benefit', 'Total_Payoff', 'Average_Payoff', 'Agent1_City_Count', 'Agent2_City_Count', 'Agent1_Timestamps', 'Agent2_Timestamps'});

% Define the file path
filePath = 'OUTPUT.csv';

% Check if the file exists and append the results if it does
if exist(filePath, 'file')
    % File exists, read existing data
    existingData = readtable(filePath);
    % Append new results
    updatedResultsTable = [existingData; resultsTable];
else
    % File does not exist, use new results directly
    updatedResultsTable = resultsTable;
end

% Save the updated table to the CSV file
writetable(updatedResultsTable, filePath, 'WriteMode', 'overwrite');

fprintf('Results including city counts, payoffs, and timestamps have been updated in %s\n', filePath);

% Differential Evolution optimization function with fixed starting cities
function [agent1_path, agent2_path, agent1_benefit, agent2_benefit, agent1_cost, agent2_cost, agent1_timestamps, agent2_timestamps] = optimize_agents_paths_DE(benefits, cost_matrix, distance_matrix, starting_cities)
    num_cities = size(distance_matrix, 1);
    population_size = 20; % Number of candidate solutions
    max_generations = 100; % Maximum number of generations
    F = 0.8; % Mutation factor
    CR = 0.9; % Crossover probability

    % Initialize population
    population = false(population_size, num_cities);
    for i = 1:population_size
        remaining_indices = setdiff(1:num_cities, starting_cities);
        population(i, remaining_indices) = rand(1, length(remaining_indices)) > 0.5;
    end

    % Ensure starting cities are correctly assigned in the population
    population(:, starting_cities(1)) = true;
    population(:, starting_cities(2)) = false;

    % Evaluate initial population
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        fitness(i) = calculate_agent_fitness(1:num_cities, population(i, :), benefits, cost_matrix, distance_matrix, starting_cities);
    end

    % Differential Evolution algorithm
    for gen = 1:max_generations
        for i = 1:population_size
            % Mutation: ensure that idxs do not include the current index i
            idxs = randperm(population_size, 3);
            while any(idxs == i)
                idxs = randperm(population_size, 3);
            end
            
            mutant = population(idxs(1), :) + F * (population(idxs(2), :) - population(idxs(3), :));
            mutant = mutant > 0.5;

            % Crossover
            trial = population(i, :);
            for j = 1:num_cities
                if rand() < CR
                    trial(j) = mutant(j);
                end
            end

            % Ensure starting cities are correctly assigned in the trial
            trial(starting_cities(1)) = true;
            trial(starting_cities(2)) = false;

            % Selection
            trial_fitness = calculate_agent_fitness(1:num_cities, trial, benefits, cost_matrix, distance_matrix, starting_cities);
            if trial_fitness > fitness(i)
                population(i, :) = trial;
                fitness(i) = trial_fitness;
            end
        end
    end

    % Get the best solution
    [~, best_idx] = max(fitness);
    best_solution = population(best_idx, :);

    % Decode the final solution to paths, ensuring starting cities are the first in each path
    remaining_indices = setdiff(1:num_cities, starting_cities);
    agent1_path = [starting_cities(1), remaining_indices(best_solution(remaining_indices))];
    agent2_path = [starting_cities(2), remaining_indices(~best_solution(remaining_indices))];

    % Calculate final benefits and costs using timestamps
    if ~isempty(agent1_path)
        [agent1_benefit, agent1_cost, agent1_timestamps] = calculate_benefit_and_cost_with_timestamps(agent1_path, agent2_path, benefits, cost_matrix, distance_matrix, starting_cities);
    else
        agent1_benefit = 0; agent1_cost = 0; agent1_timestamps = [];
    end
    if ~isempty(agent2_path)
        [agent2_benefit, agent2_cost, agent2_timestamps] = calculate_benefit_and_cost_with_timestamps(agent2_path, agent1_path, benefits, cost_matrix, distance_matrix, starting_cities);
    else
        agent2_benefit = 0; agent2_cost = 0; agent2_timestamps = [];
    end
end

% Calculate benefits, costs, and timestamps with full starting city benefits
function [total_benefit, total_cost, timestamps] = calculate_benefit_and_cost_with_timestamps(agent_path, other_agent_path, benefits, cost_matrix, distances, starting_cities)
    total_benefit = 0;
    total_cost = 0;
    timestamps = zeros(1, length(agent_path));
    if isempty(agent_path)
        return; % Skip processing for an empty path
    end
    timestamps(1) = 0; % Start from time 0
    other_timestamps = generate_timestamps(other_agent_path, distances);
    
    % Add full benefit for starting cities
    if ismember(agent_path(1), starting_cities)
        total_benefit = total_benefit + benefits(agent_path(1));
    end
    
    for i = 1:(length(agent_path) - 1)
        total_cost = total_cost + cost_matrix(agent_path(i), agent_path(i + 1));
        timestamps(i + 1) = timestamps(i) + distances(agent_path(i), agent_path(i + 1));
        
        % Determine benefit based on timestamps of the other agent
        idx_other = find(other_agent_path == agent_path(i + 1), 1);
        if ~isempty(idx_other) && other_timestamps(idx_other) == timestamps(i + 1)
            total_benefit = total_benefit + benefits(agent_path(i + 1)) / 2; % Share benefit
        elseif isempty(idx_other) || other_timestamps(idx_other) > timestamps(i + 1)
            total_benefit = total_benefit + benefits(agent_path(i + 1)); % Full benefit
        end
    end
    
    % Ensure agent_path has more than one city before calculating the return cost
    if length(agent_path) > 1
        total_cost = total_cost + cost_matrix(agent_path(end), agent_path(1));
        timestamps(end + 1) = timestamps(end) + distances(agent_path(end), agent_path(1));
    end
end

% Function to calculate the fitness of each agent using timestamps
function fitness = calculate_agent_fitness(complete_tour, division, benefits, cost_matrix, distances, starting_cities)
    % This function calculates the fitness of a given division of cities between two agents
    agent1_path = complete_tour(division == 1);
    agent2_path = complete_tour(division == 0);
    [agent1_benefit, agent1_cost] = calculate_benefit_and_cost_with_timestamps(agent1_path, agent2_path, benefits, cost_matrix, distances, starting_cities);
    [agent2_benefit, agent2_cost] = calculate_benefit_and_cost_with_timestamps(agent2_path, agent1_path, benefits, cost_matrix, distances, starting_cities);
    fitness = (agent1_benefit + agent2_benefit) - (agent1_cost + agent2_cost); % Objective function
end

% Function to generate timestamps for each agent's path
function timestamps = generate_timestamps(path, distances)
    timestamps = zeros(1, length(path));
    if isempty(path)
        return; % If the path is empty, return early
    end
    timestamps(1) = 0; % Start from time 0
    for i = 2:length(path)
        timestamps(i) = timestamps(i - 1) + distances(path(i - 1), path(i));
    end
end

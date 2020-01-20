#!/usr/bin/env th

------------------------------------------------------------------
-- This script loads the CSV file into a torch structure 
--
-- The input file is of the following format:
-- GeneID, WindowNumber,Feature1, Feature 2,....., GeneExpression
-- 
-- All values are numeric  

-- make sure you specify the number of windows for each gene in "local windows"

-- Ritambhara Singh(rs3zz@virginia.edu)
-------------------------------------------------------------------
require 'torch'
require 'math'

-- Read CSV file


-- Change file name to "test.csv" from "valid.csv" while testing
inFiles={opt.dataDir .. opt.dataset .. "/" .. "train.csv", opt.dataDir .. opt.dataset .. "/" .. "valid.csv"}

trainset={}
testset={}

for f=1,2 do

    local filePath = inFiles[f] -- Input .csv file name
    print(filePath)
    local windows = 100       -- Specify number of windows per gene

    -- Count number of rows and columns in file

    local i = 0
    for line in io.lines(filePath) do
    	if i == 0 then
      	   COLS = #line:split(',')
    	end
    	i = i + 1
    end

    --local ROWS = i-1 -- Use minus 1 if header present
    local MAXSTRLEN=100
    local ROWS = i  
    local GENES = ROWS / windows

    print("Gene Count:",GENES)
    print ("Number of entries:",ROWS)
    print("Number of features:",COLS-3)


    --Read data from CSV to tensor

    local csvFile = io.open(filePath, 'r')
    --local header = csvFile:read() -- use if header present


    local i = 0
    local j = 1
    local len

    new_data={}
    local data={}
    local label
    local geneID
    for line in csvFile:lines('*l') do
    	i=i+1
    	local l = line:split(',')
	data[i]={}
    	for key, val in ipairs(l) do
            --print(i,key,val)
	    if key==1 then
	       geneID=val
	    elseif key==COLS then
	        if tonumber(val)==0 then
		   label=2
		else
		   label=1 
		end
       	    elseif key>2 and key<=COLS-1 then
		data[i][key-2]=val
	    end
    	end
    	if i==windows then
	   table.insert(new_data,{geneID=geneID,data=torch.DoubleTensor(data),label=label})
           i=0
           j=j+1
    	end
    end
    print("Read entries!!") 
    csvFile:close()
    
    if f==1 then trainset=new_data else testset=new_data end
    
end

-------------------------End of Code -----------------------------------------

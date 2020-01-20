--[[ An auRoc class
-- Same form as the torch optim.ConfusionMatrix class
-- output is assumed to be a ranking (e.g. probability value)
-- label is assumed to be 1 or -1
Example:
    auRoc = auRoc.new()   -- new matrix
    conf:zero()                                              -- reset matrix
    for i = 1,N do
        conf:add( output, label )         -- accumulate errors
    end
    print(auRoc:calculateAuc())
]]

--local opt = require('doall')

debug.getregistry()['auRoc'] = nil 

local auRoc = torch.class("auRoc")

function auRoc:__init()
   self.target = {}
   self.pred = {}
   self.roc = 0
   self.auc = 0
end

function auRoc:add(prediction, target)
  table.insert(self.pred,prediction)
  table.insert(self.target,target)
end


function auRoc:zero()
   self.target = {}
   self.pred = {}
   self.roc = 0
   self.auc = 0
end


local function tableToTensor(table)
  local tensor = torch.Tensor(#table)
  for i = 1,#table do
    tensor[i] = table[i]
  end
  return tensor
end


local function get_rates(responses, labels)
  torch.setdefaulttensortype('torch.FloatTensor')

  responses = torch.Tensor(responses:size()):copy(responses)
  labels = torch.Tensor(labels:size()):copy(labels)

   -- assertions about the data format expected
   assert(responses:size():size() == 1, "responses should be a 1D vector")
   assert(labels:size():size() == 1 , "labels should be a 1D vector")

   -- assuming labels {-1, 1}
   local npositives = torch.sum(torch.eq(labels,  1))
   local nnegatives = torch.sum(torch.eq(labels, -1))
   local nsamples = npositives + nnegatives

   assert(nsamples == responses:size()[1], "labels should contain only -1 or 1 values")

   -- sort by response value

   local responses_sorted, indexes_sorted = torch.sort(responses,1,true)
   local labels_sorted = labels:index(1, indexes_sorted)


   local found_positives = 0
   local found_negatives = 0

   local tpr = {0} -- true pos rate
   local fpr = {0} -- false pos rate

   for i = 1,nsamples-1 do
      if labels_sorted[i] == -1 then
         found_negatives = found_negatives + 1
      else
         found_positives = found_positives + 1
      end

      table.insert(tpr, found_positives/npositives)
      table.insert(fpr, found_negatives/nnegatives)
   end

   table.insert(tpr, 1.0)
   table.insert(fpr, 1.0)

   --if opt.cuda then
       --torch.setdefaulttensortype('torch.CudaTensor')
   --end

   return tpr, fpr
end

local function find_auc(tpr,fpr)
   local area = 0.0
   for i = 2,#tpr do
      local xdiff = fpr[i] - fpr[i-1]
      local ydiff = tpr[i] - tpr[i-1]
      area = area + (xdiff * tpr[i])
   end
   return area
end


function auRoc:calculateAuc()
  local aucPredTens = tableToTensor(self.pred)
  local aucTargetTens = tableToTensor(self.target)

  local tpr = nil
  local fpr = nil

  tpr,fpr = get_rates(aucPredTens,aucTargetTens)
  self.auc = find_auc(tpr,fpr)

  return self.auc
end

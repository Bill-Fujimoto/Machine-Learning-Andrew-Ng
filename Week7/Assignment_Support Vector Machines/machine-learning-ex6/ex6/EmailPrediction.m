%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');

C = 0.1;
% This try-catch checks whether the training has been done.  If done, it saves the time of not repeating the training sequence.

try
  if ~isstruct(model)
    model = svmTrain(X, y, C, @linearKernel);
  endif
catch
  fprintf('\nTraining Linear SVM (Spam Classification)\n')
  fprintf('(this may take 1 to 2 minutes) ...\n')
  model = svmTrain(X, y, C, @linearKernel);
end_try_catch

%filename = 'Important Lutron and Nest Product Integration Update.txt';
filename = 'An Off-Wall-Street Heros Guide to Your Next Million.txt'

% Read email and Feature extraction 'x' of given email
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);

% Make spam prediction
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');


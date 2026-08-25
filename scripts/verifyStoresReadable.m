function verifyStoresReadable(storeDirectory, options)
% verifyStoresReadable - Open every Zarr v3 NWB store in a folder with nwbRead.
%
% Syntax:
%  VERIFYSTORESREADABLE(storeDirectory) reads every "*.nwb.zarr" store in
%  storeDirectory and throws if any store that is not listed as a known failure
%  could not be read.
%
%  VERIFYSTORESREADABLE(storeDirectory, Name=Value) controls which stores are
%  allowed to fail and where the summary is written.
%
% Input Arguments:
%  - storeDirectory (string) -
%    Folder containing the "*.nwb.zarr" stores to read.
%
%  - options (name-value pairs) -
%
%    - KnownFailureFile (string) -
%      Text file listing stores that are expected to fail, one name per line.
%      Blank lines and text following "#" are ignored. Default: none.
%
%    - SummaryFile (string) -
%      File to append a Markdown result table to, for use as a GitHub Actions job
%      summary. Default: the GITHUB_STEP_SUMMARY environment variable, if set.
%
% The function throws NWB:Zarr3Compat:UnexpectedResult if a store outside the
% known-failure list fails to read, or if a store on that list reads successfully
% and its entry is therefore stale.

    arguments
        storeDirectory (1,1) string {mustBeFolder}
        options.KnownFailureFile (1,1) string = ""
        options.SummaryFile (1,1) string = string(getenv("GITHUB_STEP_SUMMARY"))
    end

    knownFailures = readKnownFailures(options.KnownFailureFile);

    listing = dir(fullfile(storeDirectory, "*.nwb.zarr"));
    storeNames = sort(string({listing.name}))';
    assert(~isempty(storeNames), ...
        "NWB:Zarr3Compat:NoStores", ...
        "No *.nwb.zarr stores found in '%s'. Run the tutorials first.", storeDirectory)

    % Generate the NWB type classes once, into a scratch folder, so that each nwbRead
    % call can use "ignorecache" and skip regenerating them per store.
    classDirectory = fullfile(tempdir, "matnwb-generated-classes");
    if ~isfolder(classDirectory)
        mkdir(classDirectory)
    end
    addpath(classDirectory)
    generateCore(savedir=classDirectory)

    results = struct("Store", {}, "Passed", {}, "Message", {});
    for iStore = 1:numel(storeNames)
        storeName = storeNames(iStore);
        [passed, message] = tryRead(fullfile(storeDirectory, storeName), classDirectory);
        results(end+1) = struct("Store", storeName, "Passed", passed, "Message", message); %#ok<AGROW>
        fprintf("%-6s %s\n", statusLabel(passed, ismember(storeName, knownFailures)), storeName);
    end

    reportResults(results, knownFailures, options.SummaryFile)
end

function [passed, message] = tryRead(storePath, classDirectory)
% tryRead - Read one store, capturing the failure reason rather than throwing.
    try
        nwb = nwbRead(storePath, "ignorecache", savedir=classDirectory); %#ok<NASGU>
        passed = true;
        message = "";
    catch cause
        passed = false;
        message = flattenMessage(cause);
    end
end

function message = flattenMessage(exception)
% flattenMessage - Build a one-line summary from an exception and its first cause.
%
% MatNWB wraps the informative error (the failing property or link) as a cause of a
% generic "failed to create object" error, so the cause carries the useful detail.
    message = exception.identifier + " :: " + firstLine(exception.message);
    if ~isempty(exception.cause)
        message = message + " | " + firstLine(exception.cause{1}.message);
    end
end

function line = firstLine(text)
    lines = splitlines(strtrim(string(text)));
    line = strtrim(lines(1));
end

function label = statusLabel(passed, isKnownFailure)
    if passed && isKnownFailure
        label = "FIXED";
    elseif passed
        label = "PASS";
    elseif isKnownFailure
        label = "KNOWN";
    else
        label = "FAIL";
    end
end

function knownFailures = readKnownFailures(knownFailureFile)
% readKnownFailures - Read store names from the allowlist, ignoring comments.
    knownFailures = string.empty(0, 1);
    if knownFailureFile == "" || ~isfile(knownFailureFile)
        return
    end
    lines = splitlines(string(fileread(knownFailureFile)));
    lines = strtrim(extractBefore(lines + "#", "#"));
    knownFailures = lines(lines ~= "");
end

function reportResults(results, knownFailures, summaryFile)
% reportResults - Print the summary, write the job summary, and throw on any surprise.
    stores = [results.Store]';
    passed = [results.Passed]';
    isKnown = ismember(stores, knownFailures);

    unexpectedFailures = stores(~passed & ~isKnown);
    staleEntries = stores(passed & isKnown);

    fprintf("\n%d/%d stores read successfully (%d known failures).\n", ...
        sum(passed), numel(stores), sum(~passed & isKnown));

    if summaryFile ~= ""
        writeSummary(results, isKnown, summaryFile)
    end

    for iStore = 1:numel(unexpectedFailures)
        index = stores == unexpectedFailures(iStore);
        fprintf(2, "\nUnexpected failure: %s\n  %s\n", ...
            unexpectedFailures(iStore), results(index).Message);
    end

    if ~isempty(unexpectedFailures) && ~isempty(staleEntries)
        error("NWB:Zarr3Compat:UnexpectedResult", ...
            "%d store(s) failed unexpectedly (%s) and %d known-failure entry is now stale (%s).", ...
            numel(unexpectedFailures), strjoin(unexpectedFailures, ", "), ...
            numel(staleEntries), strjoin(staleEntries, ", "))
    elseif ~isempty(unexpectedFailures)
        error("NWB:Zarr3Compat:UnexpectedResult", ...
            "%d store(s) failed unexpectedly: %s", ...
            numel(unexpectedFailures), strjoin(unexpectedFailures, ", "))
    elseif ~isempty(staleEntries)
        error("NWB:Zarr3Compat:UnexpectedResult", ...
            "%d known-failure entry is now stale, remove it from the allowlist: %s", ...
            numel(staleEntries), strjoin(staleEntries, ", "))
    end
end

function writeSummary(results, isKnown, summaryFile)
% writeSummary - Append a Markdown table of the results for the GitHub job summary.
    fileId = fopen(summaryFile, "a");
    if fileId == -1
        warning("NWB:Zarr3Compat:SummaryUnavailable", ...
            "Could not open '%s' to write the job summary.", summaryFile)
        return
    end
    cleanup = onCleanup(@() fclose(fileId));

    fprintf(fileId, "## MatNWB read check\n\n");
    fprintf(fileId, "| Store | Result | Detail |\n| --- | --- | --- |\n");
    for iStore = 1:numel(results)
        label = statusLabel(results(iStore).Passed, isKnown(iStore));
        detail = replace(results(iStore).Message, "|", "\\|");
        fprintf(fileId, "| `%s` | %s | %s |\n", results(iStore).Store, label, detail);
    end
    fprintf(fileId, "\n`KNOWN` = listed in `ci/matnwb_known_failures.txt`. ");
    fprintf(fileId, "`FIXED` = listed there but now passing, so the entry is stale.\n");
end

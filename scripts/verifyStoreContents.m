function verifyStoreContents(storeDirectory, options)
% verifyStoreContents - Check that MatNWB reads every store the way PyNWB does.
%
% Syntax:
%  VERIFYSTORECONTENTS(storeDirectory) runs StoreContentsTest for every
%  "*.nwb.zarr" store in storeDirectory that has a manifest, and throws if a
%  check fails that is not listed as a known failure.
%
%  VERIFYSTORECONTENTS(storeDirectory, Name=Value) controls where the manifests
%  are read from, which checks may fail, and where results are written.
%
% Input Arguments:
%  - storeDirectory (string) -
%    Folder containing the "*.nwb.zarr" stores.
%
%  - options (name-value pairs) -
%
%    - ManifestDirectory (string) -
%      Folder holding the "<store>.manifest.json" files written by
%      scripts/write_read_manifests.py. Default: storeDirectory.
%
%    - KnownFailureFile (string) -
%      Text file of checks that are expected to fail, one "<store> <check>" pair
%      per line, where <check> is a StoreContentsTest method name or "*" for every
%      check of that store. Blank lines and text following "#" are ignored.
%      Default: none.
%
%    - SummaryFile (string) -
%      File to append a Markdown result table to, for use as a GitHub Actions job
%      summary. Default: the GITHUB_STEP_SUMMARY environment variable, if set.
%
%    - JUnitFile (string) -
%      JUnit XML report of every check. Default: none.
%
%    - StoreUrl (string) -
%      URL at which storeDirectory is served over HTTP. When given, each store
%      is read from "<StoreUrl>/<store>" instead of from disk; storeDirectory
%      is then only used to list the stores. Default: none.
%
%    - Title (string) -
%      Heading of the job summary table. Default: "MatNWB content check".
%
% The function throws NWB:Zarr3Compat:UnexpectedResult if a check outside the
% known-failure list fails, or if a check on that list passes and its entry is
% therefore stale.

    arguments
        storeDirectory (1,1) string {mustBeFolder}
        options.ManifestDirectory (1,1) string = storeDirectory
        options.KnownFailureFile (1,1) string = ""
        options.SummaryFile (1,1) string = string(getenv("GITHUB_STEP_SUMMARY"))
        options.JUnitFile (1,1) string = ""
        options.StoreUrl (1,1) string = ""
        options.Title (1,1) string = "MatNWB content check"
    end

    import matlab.unittest.TestRunner
    import matlab.unittest.TestSuite
    import matlab.unittest.parameters.Parameter
    import matlab.unittest.plugins.XMLPlugin

    knownFailures = readKnownFailures(options.KnownFailureFile);
    storeNames = storesWithManifests(storeDirectory, options.ManifestDirectory);
    classDirectory = prepareClassDirectory();

    % Parameter names must be valid identifiers, so each store is keyed by a
    % mangled name and its real name is recovered from the value.
    storeParameters = struct();
    for storeName = storeNames'
        if options.StoreUrl == ""
            storePath = fullfile(storeDirectory, storeName);
        else
            storePath = strip(options.StoreUrl, "right", "/") + "/" + storeName;
        end
        storeParameters.(matlab.lang.makeValidName(storeName)) = struct( ...
            "StorePath", storePath, ...
            "ManifestPath", fullfile(options.ManifestDirectory, storeName + ".manifest.json"), ...
            "ClassDirectory", classDirectory);
    end
    parameter = Parameter.fromData("Store", storeParameters);
    suite = TestSuite.fromClass(?StoreContentsTest, ExternalParameters=parameter);

    runner = TestRunner.withTextOutput();
    if options.JUnitFile ~= ""
        reportFolder = fileparts(options.JUnitFile);
        if reportFolder ~= "" && ~isfolder(reportFolder)
            mkdir(reportFolder)
        end
        runner.addPlugin(XMLPlugin.producingJUnitFormat(options.JUnitFile));
    end
    results = runner.run(suite);

    outcomes = classifyResults(suite, results, storeParameters, knownFailures);
    reportOutcomes(outcomes, knownFailures, options.SummaryFile, options.Title)
end

function storeNames = storesWithManifests(storeDirectory, manifestDirectory)
    listing = dir(fullfile(storeDirectory, "*.nwb.zarr"));
    storeNames = sort(string({listing.name}))';
    hasManifest = arrayfun(@(name) isfile(fullfile(manifestDirectory, name + ".manifest.json")), storeNames);
    missing = storeNames(~hasManifest);
    assert(isempty(missing), "NWB:Zarr3Compat:MissingManifest", ...
        "No manifest for %s in '%s'. Run scripts/write_read_manifests.py first.", ...
        strjoin(missing, ", "), manifestDirectory)
    assert(~isempty(storeNames), "NWB:Zarr3Compat:NoStores", ...
        "No *.nwb.zarr stores found in '%s'. Run the tutorials first.", storeDirectory)
end

function classDirectory = prepareClassDirectory()
% prepareClassDirectory - Empty folder for the type classes nwbRead generates.
%
% Several stores embed extension schemas whose classes exist only once they are
% generated from the store being read. Starting from an empty folder keeps classes
% generated by earlier work from making a store look readable.
    classDirectory = fullfile(tempdir, "matnwb-content-check-classes");
    if isfolder(classDirectory)
        rmdir(classDirectory, "s")
    end
    mkdir(classDirectory)
    addpath(classDirectory)
    generateCore(savedir=classDirectory)
end

function outcomes = classifyResults(suite, results, storeParameters, knownFailures)
% classifyResults - One row per store and check, with its pass/fail state.
    parameterNames = string(fieldnames(storeParameters));
    storeOf = dictionary();
    for name = parameterNames'
        [~, storeName, extension] = fileparts(storeParameters.(name).StorePath);
        storeOf(name) = storeName + extension;
    end

    numResults = numel(results);
    store = strings(numResults, 1);
    check = strings(numResults, 1);
    for iResult = 1:numResults
        parameterization = suite(iResult).Parameterization;
        store(iResult) = storeOf(string(parameterization(1).Name));
        check(iResult) = string(suite(iResult).ProcedureName);
    end
    passed = [results.Passed]';
    isKnown = ismember(store + " " + check, knownFailures) | ismember(store + " *", knownFailures);
    outcomes = table(store, check, passed, isKnown, 'VariableNames', ["Store", "Check", "Passed", "IsKnown"]);
end

function reportOutcomes(outcomes, knownFailures, summaryFile, title)
    unexpected = outcomes(~outcomes.Passed & ~outcomes.IsKnown, :);
    stale = staleEntries(outcomes, knownFailures);

    fprintf("\n%d/%d checks passed (%d known failures).\n", sum(outcomes.Passed), ...
        height(outcomes), sum(~outcomes.Passed & outcomes.IsKnown));

    if summaryFile ~= ""
        writeSummary(outcomes, summaryFile, title)
    end

    messages = strings(0, 1);
    if ~isempty(unexpected)
        messages(end+1) = sprintf("%d check(s) failed unexpectedly: %s", height(unexpected), ...
            strjoin(unexpected.Store + " " + unexpected.Check, ", "));
    end
    if ~isempty(stale)
        messages(end+1) = sprintf("%d known-failure entry is now stale, remove it: %s", ...
            numel(stale), strjoin(stale, ", "));
    end
    if ~isempty(messages)
        error("NWB:Zarr3Compat:UnexpectedResult", "%s", strjoin(messages, " | "))
    end
end

function stale = staleEntries(outcomes, knownFailures)
% staleEntries - Known-failure entries whose checks all pass.
    stale = strings(0, 1);
    for entry = reshape(knownFailures, 1, [])
        parts = split(entry);
        if parts(2) == "*"
            covered = outcomes.Store == parts(1);
        else
            covered = outcomes.Store == parts(1) & outcomes.Check == parts(2);
        end
        if ~any(covered & ~outcomes.Passed)
            stale(end+1) = entry; %#ok<AGROW>
        end
    end
end

function knownFailures = readKnownFailures(knownFailureFile)
% readKnownFailures - "<store> <check>" pairs from the allowlist, ignoring comments.
    knownFailures = strings(0, 1);
    if knownFailureFile == "" || ~isfile(knownFailureFile)
        return
    end
    lines = splitlines(string(fileread(knownFailureFile)));
    lines = strtrim(extractBefore(lines + "#", "#"));
    lines = lines(lines ~= "");
    knownFailures = join(split(lines(:)), " ", 2);
    knownFailures = reshape(knownFailures, [], 1);
end

function writeSummary(outcomes, summaryFile, title)
% writeSummary - Append a store x check Markdown table for the GitHub job summary.
    fileId = fopen(summaryFile, "a");
    if fileId == -1
        warning("NWB:Zarr3Compat:SummaryUnavailable", ...
            "Could not open '%s' to write the job summary.", summaryFile)
        return
    end
    cleanup = onCleanup(@() fclose(fileId));

    checks = unique(outcomes.Check, "stable");
    stores = unique(outcomes.Store);
    fprintf(fileId, "## %s\n\n", title);
    fprintf(fileId, "| Store | %s |\n", strjoin("`" + checks + "`", " | "));
    fprintf(fileId, "| --- |%s\n", strjoin(repmat(" --- |", 1, numel(checks)), ""));
    for store = stores'
        labels = strings(1, numel(checks));
        for iCheck = 1:numel(checks)
            row = outcomes(outcomes.Store == store & outcomes.Check == checks(iCheck), :);
            labels(iCheck) = statusLabel(row.Passed, row.IsKnown);
        end
        fprintf(fileId, "| `%s` | %s |\n", store, strjoin(labels, " | "));
    end
    fprintf(fileId, "\n`KNOWN` = listed in `ci/matnwb_content_known_failures.txt`. ");
    fprintf(fileId, "`FIXED` = listed there but now passing, so the entry is stale. ");
    fprintf(fileId, "Failure details are in the job log.\n");
end

function label = statusLabel(passed, isKnown)
    if passed && isKnown
        label = "FIXED";
    elseif passed
        label = "PASS";
    elseif isKnown
        label = "KNOWN";
    else
        label = "FAIL";
    end
end

classdef StoreContentsTest < matlab.unittest.TestCase
% StoreContentsTest - Compare what MatNWB reads from a store with what PyNWB read.
%
% Each store is read once with nwbRead, and every check compares the result with
% the store's manifest, written by scripts/write_read_manifests.py from what PyNWB
% read. The manifest keys every entry by HDF5-style path, which NwbFile.resolve
% follows to the matching MatNWB object or property.
%
% Run through verifyStoreContents, which supplies the Store parameter. Each value
% is a struct with fields StorePath, ManifestPath and ClassDirectory.

    properties (ClassSetupParameter)
        % Replaced by verifyStoreContents through external parameters.
        Store = struct("placeholder", struct( ...
            "StorePath", "", "ManifestPath", "", "ClassDirectory", ""))
    end

    properties (Constant, Access = private)
        % A dataset at least this large must be read lazily, as a DataStub.
        MinLazyElements = 1000
        % Datasets up to this size are also loaded whole to compare their sum.
        MaxFullLoadElements = 1e6
        RelativeTolerance = 1e-6
        % float32 data round-trips through float64 in the manifest.
        SingleRelativeTolerance = 1e-5
        PosixTolerance = 1e-3
    end

    properties (Access = private)
        Nwb
        Manifest
    end

    methods (TestClassSetup)
        function readStore(testCase, Store)
            testCase.Manifest = jsondecode(fileread(Store.ManifestPath));
            % An assertion in class setup fails every check of this store with the
            % same cause, and the run moves on to the next store.
            try
                testCase.Nwb = nwbRead(Store.StorePath, savedir=Store.ClassDirectory);
            catch exception
                testCase.assertFail("nwbRead failed: " + exception.identifier ...
                    + ": " + firstLine(exception.message));
            end
        end
    end

    methods (Test)
        function resolvesTypedObjects(testCase)
            for entry = entries(testCase.Manifest.objects)
                label = "object " + entry.path;
                object = testCase.resolveOrFail(entry.path, label);
                if isempty(object)
                    continue
                end
                expectedClass = "types." + replace(entry.namespace, "-", "_") ...
                    + "." + entry.neurodata_type;
                testCase.verifyTrue(isa(object, expectedClass), ...
                    label + ": expected " + expectedClass + ", got " + class(object));
                if isprop(object, "object_id")
                    testCase.verifyEqual(string(object.object_id), string(entry.object_id), label);
                end
            end
        end

        function resolvesLinks(testCase)
            for entry = entries(testCase.Manifest.links)
                label = "link " + entry.path + " -> " + entry.target.path;
                link = testCase.resolveOrFail(entry.path, label);
                if isempty(link)
                    continue
                end
                if entry.external
                    testCase.verifyClass(link, "types.untyped.ExternalLink", label);
                    if ~isa(link, "types.untyped.ExternalLink")
                        continue
                    end
                    testCase.verifyEqual(string(link.path), string(entry.target.path), label);
                    [~, name, extension] = fileparts(link.filename);
                    testCase.verifyEqual(string(name) + extension, string(entry.source), label);
                    target = testCase.callOrFail(@() link.deref(), label + " (deref)");
                else
                    testCase.verifyClass(link, "types.untyped.SoftLink", label);
                    if ~isa(link, "types.untyped.SoftLink")
                        continue
                    end
                    testCase.verifyEqual(string(link.path), string(entry.target.path), label);
                    target = testCase.callOrFail(@() link.deref(testCase.Nwb), label + " (deref)");
                end
                testCase.verifyObjectId(target, entry.target, label);
            end
        end

        function resolvesReferences(testCase)
            for entry = entries(testCase.Manifest.references)
                label = entry.kind + " reference " + entry.path;
                value = testCase.resolveOrFail(entry.path, label);
                if isempty(value)
                    continue
                end
                value = unwrapData(value);
                if isa(value, "types.untyped.DataStub")
                    value = testCase.callOrFail(@() value.load(), label + " (load)");
                end
                testCase.verifyClass(value, "types.untyped.ObjectView", label);
                if ~isa(value, "types.untyped.ObjectView")
                    continue
                end
                targets = entries(entry.targets);
                testCase.verifyNumElements(value, numel(targets), label);
                for iTarget = 1:min(numel(value), numel(targets))
                    testCase.verifyReference(value(iTarget), targets(iTarget), ...
                        label + "(" + iTarget + ")");
                end
            end
        end

        function readsDatasetShapes(testCase)
            for entry = entries(testCase.Manifest.datasets)
                label = "dataset " + entry.path;
                value = testCase.resolveDatasetOrFail(entry, label);
                if isempty(value) && entry.count > 0
                    continue
                end
                expectedDims = matnwbDims(entry.shape);
                if isa(value, "types.untyped.DataStub")
                    testCase.verifyEqual(double(value.dims), expectedDims, label);
                elseif ~isempty(entry.shape)
                    testCase.verifyEqual(numel(value), entry.count, label);
                end
            end
        end

        function readsDatasetSamples(testCase)
        % Each sample is read through indexing, which for a DataStub is a partial
        % read of the one element.
            for entry = entries(testCase.Manifest.datasets)
                label = "dataset " + entry.path;
                value = testCase.resolveDatasetOrFail(entry, label);
                if isempty(value)
                    continue
                end
                for sample = entries(entry.samples)
                    sampleLabel = label + " at [" + strjoin(string(sample.index), ",") + "]";
                    [actual, isRead] = testCase.callOrFail( ...
                        @() readElement(value, sample.index, numel(entry.shape)), sampleLabel);
                    if isRead
                        testCase.verifyValue(actual, sample.value, entry.kind, sampleLabel);
                    end
                end
            end
        end

        function readsDatasetsInFull(testCase)
            for entry = entries(testCase.Manifest.datasets)
                if ~isfield(entry, "sum") || isempty(entry.sum) ...
                        || entry.count > testCase.MaxFullLoadElements
                    continue
                end
                label = "dataset " + entry.path + " (full load)";
                value = testCase.resolveDatasetOrFail(entry, label);
                if isa(value, "types.untyped.DataStub")
                    [value, isLoaded] = testCase.callOrFail(@() value.load(), label);
                    if ~isLoaded
                        continue
                    end
                end
                testCase.verifyNumElements(value, entry.count, label);
                values = double(value(:));
                testCase.verifyValue(sum(values(isfinite(values))), entry.sum, "numeric", label);
            end
        end

        function keepsLargeDatasetsLazy(testCase)
            for entry = entries(testCase.Manifest.datasets)
                if entry.count < testCase.MinLazyElements
                    continue
                end
                label = "dataset " + entry.path + " (" + entry.count + " elements)";
                value = testCase.resolveDatasetOrFail(entry, label);
                testCase.verifyClass(value, "types.untyped.DataStub", label);
            end
        end

        function readsCompoundDatasets(testCase)
            for entry = entries(testCase.Manifest.compounds)
                label = "compound " + entry.path;
                value = testCase.resolveDatasetOrFail(entry, label);
                if isempty(value)
                    continue
                end
                if isa(value, "types.untyped.DataStub")
                    value = testCase.callOrFail(@() value.load(), label + " (load)");
                end
                if istable(value)
                    value = table2struct(value, "ToScalar", true);
                end
                testCase.verifyClass(value, "struct", label);
                if ~isstruct(value)
                    continue
                end
                testCase.verifyEqual(sort(string(fieldnames(value))), ...
                    sort(string(entry.fields)), label);
                for record = entries(entry.records)
                    for field = reshape(string(entry.fields), 1, [])
                        recordLabel = label + "(" + (record.index + 1) + ")." + field;
                        if ~isfield(value, field)
                            continue
                        end
                        actual = value.(field)(record.index + 1);
                        expected = record.values.(matlab.lang.makeValidName(field));
                        if isstruct(expected) && isfield(expected, "reference")
                            testCase.verifyReference(actual, expected.reference, recordLabel);
                        else
                            testCase.verifyValue(actual, expected, kindOf(expected), recordLabel);
                        end
                    end
                end
            end
        end

        function readsAttributes(testCase)
            for entry = entries(testCase.Manifest.attributes)
                label = "attribute " + entry.path;
                value = testCase.resolveOrFail(entry.path, label, isempty(entry.value));
                if isempty(value) && ~isempty(entry.value)
                    continue
                end
                testCase.verifyValue(value, entry.value, entry.kind, label);
            end
        end

        function readsTables(testCase)
            for entry = entries(testCase.Manifest.tables)
                label = "table " + entry.path;
                dynamicTable = testCase.resolveOrFail(entry.path, label);
                if isempty(dynamicTable)
                    continue
                end
                testCase.verifyEqual(textRow(dynamicTable.colnames), textRow(entry.colnames), ...
                    label + " colnames");
                if isempty(entry.colnames)
                    entry.colnames = {};
                end
                ids = loadIfStub(dynamicTable.id.data);
                testCase.verifyEqual(double(ids(:)), double(entry.ids(:)), label + " ids");

                columnKinds = entry.column_kinds;
                for row = entries(entry.rows)
                    rowLabel = label + " row " + (row.index + 1);
                    rowTable = testCase.callOrFail( ...
                        @() dynamicTable.getRow(row.index + 1), rowLabel + " (getRow)");
                    if ~istable(rowTable)
                        continue
                    end
                    for column = reshape(string(fieldnames(row.values)), 1, [])
                        columnName = string(entry.colnames(strcmp(matlab.lang.makeValidName(entry.colnames), column)));
                        columnLabel = rowLabel + " column " + columnName;
                        testCase.verifyTrue(any(strcmp(rowTable.Properties.VariableNames, columnName)), ...
                            columnLabel + " is missing from getRow");
                        if ~any(strcmp(rowTable.Properties.VariableNames, columnName))
                            continue
                        end
                        actual = rowTable.(columnName);
                        expected = row.values.(column);
                        testCase.verifyColumnValue(actual, expected, ...
                            columnKinds.(column), columnLabel);
                    end
                end
            end
        end
    end

    methods (Access = private)
        function value = resolveOrFail(testCase, path, label, isEmptyExpected)
        % resolveOrFail - Resolve a path, recording an error as a verification failure.
        %
        % NwbFile.resolve reports a property that holds an empty value as an
        % unresolved path, so that error is accepted when the expected value is
        % empty.
            arguments
                testCase
                path (1,1) string
                label (1,1) string
                isEmptyExpected (1,1) logical = false
            end
            value = [];
            if path == "/"
                value = testCase.Nwb;
                return
            end
            try
                value = testCase.Nwb.resolve(char(path));
            catch exception
                if ~(isEmptyExpected && exception.identifier == "NWB:IO:UnresolvedPath")
                    testCase.verifyFail(label + " (resolve) threw " + exception.identifier ...
                        + ": " + firstLine(exception.message));
                end
            end
        end

        function value = resolveDatasetOrFail(testCase, entry, label)
            value = unwrapData(testCase.resolveOrFail(entry.path, label, entry.count == 0));
        end

        function [value, isDone] = callOrFail(testCase, operation, label)
        % callOrFail - Run operation, recording an error as a verification failure.
            value = [];
            isDone = false;
            try
                value = operation();
                isDone = true;
            catch exception
                testCase.verifyFail(label + " threw " + exception.identifier + ": " ...
                    + firstLine(exception.message));
            end
        end

        function verifyObjectId(testCase, object, expectedTarget, label)
            if isempty(expectedTarget.object_id) || isempty(object)
                return
            end
            testCase.verifyTrue(isprop(object, "object_id") ...
                && string(object.object_id) == string(expectedTarget.object_id), ...
                label + " resolves to a different object");
        end

        function verifyReference(testCase, objectView, expectedTarget, label)
            testCase.verifyClass(objectView, "types.untyped.ObjectView", label);
            if ~isa(objectView, "types.untyped.ObjectView")
                return
            end
            if ~isempty(expectedTarget.path)
                testCase.verifyEqual(string(objectView.path), string(expectedTarget.path), label);
            end
            target = testCase.callOrFail(@() objectView.refresh(testCase.Nwb), label + " (refresh)");
            testCase.verifyObjectId(target, expectedTarget, label);
        end

        function verifyColumnValue(testCase, actual, expected, columnKind, label)
            if iscell(actual) && isscalar(actual)
                actual = actual{1};
            end
            switch columnKind
                case "reference"
                    testCase.verifyClass(actual, "types.untyped.ObjectView", label);
                    if isa(actual, "types.untyped.ObjectView")
                        target = testCase.callOrFail(@() actual.refresh(testCase.Nwb), label);
                        testCase.verifyObjectId(target, struct("object_id", expected.object_id), label);
                    end
                case {"ragged", "region", "plain"}
                    % A row value of rank 2 or more has its axes reversed, as
                    % MatNWB reverses those of the whole dataset.
                    if isnumeric(expected) && ~isvector(expected) && ~isempty(actual)
                        actual = permute(actual, ndims(actual):-1:1);
                    end
                    testCase.verifyValue(actual, expected, kindOf(expected), label);
                otherwise
                    testCase.verifyFail(label + ": unknown column kind " + columnKind);
            end
        end

        function verifyValue(testCase, actual, expected, kind, label)
            try
                [isMatch, description] = valuesMatch(actual, expected, kind, testCase);
            catch exception
                isMatch = false;
                description = "comparison threw " + exception.identifier + ": " ...
                    + firstLine(exception.message);
            end
            testCase.verifyTrue(isMatch, label + ": " + description);
        end
    end
end

function list = entries(manifestArray)
% entries - Manifest array as a 1xN struct array, whatever shape jsondecode gave it.
%
% jsondecode returns a struct array for a list of objects with the same fields and
% [] for an empty list. scripts/write_read_manifests.py writes every object of a
% list with the same fields, so a cell array (fields that differ) is an error.
    if isempty(manifestArray)
        list = struct([]);
    elseif iscell(manifestArray)
        error("NWB:Zarr3Compat:InvalidManifest", ...
            "Manifest list entries must all have the same fields.")
    else
        list = reshape(manifestArray, 1, []);
    end
end

function text = textRow(value)
% textRow - Text as a string row; any empty value is the empty list of names.
    if isempty(value)
        text = strings(1, 0);
    else
        text = reshape(string(value), 1, []);
    end
end

function value = unwrapData(value)
% unwrapData - The data of a typed dataset (VectorData, ElementIdentifiers, ...).
    if isa(value, "types.untyped.MetaClass") && isprop(value, "data")
        value = value.data;
    end
end

function value = loadIfStub(value)
    if isa(value, "types.untyped.DataStub")
        value = value.load();
    end
end

function dims = matnwbDims(shape)
% matnwbDims - MatNWB dims of a dataset with the given numpy (row-major) shape.
    shape = reshape(double(shape), 1, []);
    if numel(shape) >= 2
        dims = fliplr(shape);
    else
        dims = shape;
    end
end

function element = readElement(value, index, rank)
% readElement - One element at a zero-based numpy index, through MATLAB indexing.
%
% MatNWB reverses the axes of a dataset of rank 2 or more, so the numpy index is
% reversed as well.
    if rank == 0
        element = value;
        if isa(element, "types.untyped.DataStub")
            element = element.load();
        end
        return
    end
    subscripts = num2cell(reshape(double(index), 1, []) + 1);
    if rank >= 2
        subscripts = fliplr(subscripts);
    end
    element = value(subscripts{:});
end

function kind = kindOf(expected)
    if iscell(expected)
        expected = [expected{:}];
    end
    if islogical(expected)
        kind = "bool";
    elseif isnumeric(expected)
        kind = "numeric";
    elseif isstruct(expected)
        kind = "struct";
    else
        kind = "text";
    end
end

function [isMatch, description] = valuesMatch(actual, expected, kind, testCase)
% valuesMatch - Compare a MatNWB value with a manifest value of the given kind.
    if iscell(actual) && isscalar(actual) && ~iscellstr(actual) %#ok<ISCLSTR> unwrap only non-text cells
        actual = actual{1};
    end
    expectedValues = expectedList(expected);
    switch kind
        case "datetime"
            if isdatetime(actual)
                actualValues = posixtime(actual(:));
            else
                actualValues = posixtime(datetime(string(actual(:)), "TimeZone", "UTC"));
            end
            expectedValues = double([expectedValues{:}]');
            isMatch = numel(actualValues) == numel(expectedValues) ...
                && all(abs(actualValues - expectedValues) <= testCase.PosixTolerance);
        case {"numeric", "bool"}
            if ~(isnumeric(actual) || islogical(actual))
                isMatch = false;
                description = "expected a number, got " + class(actual);
                return
            end
            actualValues = double(actual(:));
            expectedValues = cellfun(@toNumber, expectedValues(:));
            tolerance = testCase.RelativeTolerance;
            if isa(actual, "single")
                tolerance = testCase.SingleRelativeTolerance;
            end
            isMatch = numel(actualValues) == numel(expectedValues) && all( ...
                (isnan(actualValues) & isnan(expectedValues)) ...
                | actualValues == expectedValues ...
                | abs(actualValues - expectedValues) <= tolerance*max(1, abs(expectedValues)));
        case "text"
            if isnumeric(actual) && isempty(actual)
                % MatNWB reads an empty text attribute back as [].
                actual = "";
            elseif ischar(actual)
                actual = string(actual);
            end
            actualValues = string(actual(:));
            expectedValues = string(expectedValues(:));
            isMatch = isequal(actualValues, expectedValues);
        otherwise
            isMatch = false;
    end
    description = "expected " + describe(expected) + ", got " + describe(actual);
end

function values = expectedList(expected)
    if iscell(expected)
        values = expected(:);
    elseif ischar(expected) || (isstring(expected) && isscalar(expected))
        values = {expected};
    else
        values = num2cell(expected(:));
    end
end

function number = toNumber(value)
% toNumber - Manifest number; non-finite values are written as text.
    if ischar(value) || isstring(value)
        number = str2double(value);
    else
        number = double(value);
    end
end

function text = describe(value)
    try
        if isstring(value) || ischar(value) || isnumeric(value) || islogical(value)
            text = "[" + strjoin(string(value(:)'), ", ") + "] (" + class(value) + ")";
        elseif iscell(value)
            text = "{" + strjoin(cellfun(@(v) describe(v), value(:)'), "; ") + "}";
        else
            text = "<" + class(value) + ">";
        end
    catch
        text = "<" + class(value) + ">";
    end
    if strlength(text) > 200
        text = extractBefore(text, 200) + "...";
    end
end

function line = firstLine(text)
    lines = splitlines(strtrim(string(text)));
    line = strtrim(lines(1));
end

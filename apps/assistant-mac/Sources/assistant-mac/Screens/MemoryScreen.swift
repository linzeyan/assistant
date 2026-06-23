import SwiftUI

struct MemoryScreen: View {
    @EnvironmentObject var controller: BackendController
    @State private var entries: [MemoryEntryDTO] = []
    @State private var query: String = ""
    @State private var note: String?

    // Editor sheet (new or edit) state.
    @State private var showEditor = false
    @State private var editingId: String?  // nil = creating a new memory
    @State private var draftContent = ""
    @State private var draftTags = ""
    @State private var editorError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Memory").font(.title2).bold()
                Spacer()
                Button { newMemory() } label: { Image(systemName: "plus") }
                    .help("New memory")
                TextField("Search…", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
                    .onSubmit { Task { await search() } }
                Button { Task { await search() } } label: { Image(systemName: "magnifyingglass") }
                Button { Task { await loadAll() } } label: { Image(systemName: "arrow.clockwise") }
            }
            .padding(8)

            if let note {
                Text(note).font(.caption).foregroundStyle(.secondary).padding(.horizontal, 8)
            }

            if entries.isEmpty {
                ContentUnavailableView("No memories", systemImage: "tray")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(entries) { entry in row(entry) }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .task { await loadAll() }
        .sheet(isPresented: $showEditor) { editorSheet }
    }

    private func row(_ entry: MemoryEntryDTO) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text(entry.content).textSelection(.enabled)
                if !entry.tags.isEmpty {
                    Text(entry.tags.joined(separator: ", "))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Menu {
                Button("Edit") { edit(entry) }
                Button("Delete", role: .destructive) { Task { await delete(entry.id) } }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
        .contextMenu {
            Button("Edit") { edit(entry) }
            Button("Delete", role: .destructive) { Task { await delete(entry.id) } }
        }
    }

    private var editorSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(editingId == nil ? "New memory" : "Edit memory").font(.headline)
            Text("Content").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $draftContent)
                .font(.body)
                .frame(minHeight: 140)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.secondary.opacity(0.4)))
            Text("Tags (comma-separated)").font(.caption).foregroundStyle(.secondary)
            TextField("pref, style", text: $draftTags).textFieldStyle(.roundedBorder)
            if let editorError {
                Text(editorError).foregroundStyle(.red).font(.caption)
            }
            HStack {
                Spacer()
                Button("Cancel") { showEditor = false }
                Button("Save") { Task { await saveEditor() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(draftContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding()
        .frame(width: 480, height: 340)
    }

    // MARK: - Actions

    private func newMemory() {
        editingId = nil
        draftContent = ""; draftTags = ""
        editorError = nil
        showEditor = true
    }

    private func edit(_ entry: MemoryEntryDTO) {
        editingId = entry.id
        draftContent = entry.content
        draftTags = entry.tags.joined(separator: ", ")
        editorError = nil
        showEditor = true
    }

    private func saveEditor() async {
        let content = draftContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let tags = draftTags
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        do {
            if let id = editingId {
                try await controller.client.updateMemory(id, content: content, tags: tags)
            } else {
                try await controller.client.createMemory(content: content, tags: tags)
            }
            showEditor = false
            await loadAll()
        } catch {
            editorError = "\(error)"
        }
    }

    private func delete(_ id: String) async {
        do {
            try await controller.client.deleteMemory(id)
            await loadAll()
        } catch {
            note = "Delete failed: \(error)"
        }
    }

    private func loadAll() async {
        entries = (try? await controller.client.memory().memories) ?? []
    }

    private func search() async {
        if query.trimmingCharacters(in: .whitespaces).isEmpty {
            await loadAll()
            return
        }
        entries = (try? await controller.client.searchMemory(query).results) ?? []
    }
}

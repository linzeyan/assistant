import AppKit
import SwiftUI

struct SkillsScreen: View {
    @EnvironmentObject var controller: BackendController
    @State private var skills: [SkillDTO] = []
    @State private var selectedName: String?
    @State private var detail: SkillBodyDTO?
    @State private var note: String?

    // Editor sheet (new or edit) state.
    @State private var showEditor = false
    @State private var editingNew = false
    @State private var draftName = ""
    @State private var draftDesc = ""
    @State private var draftBody = ""
    @State private var editorError: String?

    var body: some View {
        HSplitView {
            listPane.frame(minWidth: 280)
            detailPane
        }
        .task { await load() }
        .sheet(isPresented: $showEditor) { editorSheet }
    }

    // MARK: - List

    private var listPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Skills").font(.title2).bold()
                Spacer()
                Button { newSkill() } label: { Image(systemName: "plus") }
                    .help("New skill")
                Button { importSkill() } label: { Image(systemName: "square.and.arrow.down") }
                    .help("Import a SKILL.md file")
                Button("Reload") { Task { await reload() } }
                Button { Task { await load() } } label: { Image(systemName: "arrow.clockwise") }
            }
            .padding(8)

            if let note {
                Text(note).font(.caption).foregroundStyle(.secondary).padding(.horizontal, 8)
            }

            List(skills, selection: $selectedName) { skill in
                HStack {
                    VStack(alignment: .leading) {
                        Text(skill.name)
                        Text(skill.description).font(.caption).foregroundStyle(.secondary)
                    }
                    if skill.editable != true {
                        Spacer()
                        Image(systemName: "shippingbox").font(.caption).foregroundStyle(.tertiary)
                            .help("Bundled — editing creates your own copy")
                    }
                }
                .tag(skill.name)
            }
            .onChange(of: selectedName) { _, name in
                Task { await loadDetail(name) }
            }
        }
    }

    // MARK: - Detail

    private var detailPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let detail {
                HStack {
                    Text(detail.name).font(.headline)
                    if detail.editable != true {
                        Text("bundled").font(.caption)
                            .foregroundStyle(.secondary)
                            .help("Editing creates your own copy; the original is kept.")
                    }
                    Spacer()
                    // Edit is always available — editing a bundled skill is copy-on-write.
                    Button("Edit") { editSkill(detail) }
                    // Delete only applies to a user copy (reverts a bundled skill to original).
                    if detail.editable == true {
                        Button("Delete", role: .destructive) { Task { await deleteSkill(detail.name) } }
                    }
                }
                .padding(8)
                Divider()
                ScrollView {
                    Text(detail.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
            } else {
                ContentUnavailableView(
                    "No skill selected",
                    systemImage: "sparkles",
                    description: Text("Select a skill, or add one with + / import.")
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Editor sheet

    private var editorSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(editingNew ? "New skill" : "Edit \(draftName)").font(.headline)
            if editingNew {
                TextField("name (e.g. summarize-go)", text: $draftName)
                    .textFieldStyle(.roundedBorder).autocorrectionDisabled()
            }
            TextField("description", text: $draftDesc).textFieldStyle(.roundedBorder)
            Text("Body (markdown)").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $draftBody)
                .font(.body.monospaced())
                .frame(minHeight: 240)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.secondary.opacity(0.4)))
            if let editorError {
                Text(editorError).foregroundStyle(.red).font(.caption)
            }
            HStack {
                Spacer()
                Button("Cancel") { showEditor = false }
                Button(editingNew ? "Create" : "Save") { Task { await saveEditor() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(draftBody.trimmingCharacters(in: .whitespaces).isEmpty
                        || (editingNew && draftName.trimmingCharacters(in: .whitespaces).isEmpty))
            }
        }
        .padding()
        .frame(width: 580, height: 480)
    }

    // MARK: - Actions

    private func newSkill() {
        editingNew = true
        draftName = ""; draftDesc = ""; draftBody = ""
        editorError = nil
        showEditor = true
    }

    private func editSkill(_ s: SkillBodyDTO) {
        editingNew = false
        draftName = s.name
        draftDesc = s.description ?? ""
        draftBody = s.body
        editorError = nil
        showEditor = true
    }

    private func saveEditor() async {
        do {
            if editingNew {
                try await controller.client.createSkill(
                    name: draftName, description: draftDesc, body: draftBody
                )
            } else {
                try await controller.client.updateSkill(
                    draftName, description: draftDesc, body: draftBody
                )
            }
            showEditor = false
            await load()
            let slug = draftName.lowercased().replacingOccurrences(of: " ", with: "-")
            selectedName = skills.first { $0.name == slug }?.name ?? selectedName
            await loadDetail(selectedName)
        } catch {
            editorError = "\(error)"
        }
    }

    private func deleteSkill(_ name: String) async {
        do {
            try await controller.client.deleteSkill(name)
            selectedName = nil
            detail = nil
            await load()
        } catch {
            note = "Delete failed: \(error)"
        }
    }

    private func importSkill() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.init(filenameExtension: "md")].compactMap { $0 }
        panel.prompt = "Import"
        guard panel.runModal() == .OK, let url = panel.url,
            let content = try? String(contentsOf: url, encoding: .utf8)
        else { return }
        Task {
            do {
                try await controller.client.importSkill(content: content)
                await load()
                note = "Imported \(url.lastPathComponent)"
            } catch {
                note = "Import failed: \(error)"
            }
        }
    }

    private func load() async {
        skills = (try? await controller.client.skills().skills) ?? []
    }

    private func loadDetail(_ name: String?) async {
        guard let name else { detail = nil; return }
        detail = try? await controller.client.skill(name)
    }

    private func reload() async {
        if let result = try? await controller.client.reloadSkills() {
            note = "added \(result.added.count), removed \(result.removed.count), total \(result.total)"
        }
        await load()
    }
}

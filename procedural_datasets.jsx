import { useState } from "react";

const datasets = [
  // Tier 1: Hierarchical Procedural Step + Error Annotations (Gold Standard)
  {
    name: "Ego4D Goal-Step",
    venue: "NeurIPS 2023 Spotlight",
    domain: "Kitchen / Daily",
    view: "Egocentric",
    hours: 430,
    videos: 7353,
    stepSegments: "48K",
    hierarchy: "Goal → Step → Substep",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Partial",
    license: "Ego4D License",
    link: "https://ego4d-data.org/",
    notes: "Largest egocentric procedural dataset. 319 goals, 514 steps. Active CVPR 2026 challenge. Step completion status + relevance annotations.",
    tier: 1,
  },
  {
    name: "CaptainCook4D",
    venue: "NeurIPS 2024 D&B",
    domain: "Kitchen",
    view: "Egocentric",
    hours: 94.5,
    videos: 384,
    stepSegments: "5,300",
    hierarchy: "Recipe → Steps",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "Yes (baseline)",
    license: "Research",
    link: "https://captaincook4d.github.io/",
    notes: "6 error categories (technique, measurement, ordering, timing, temperature, preparation). Task graph representations. 164 correct + 220 error videos.",
    tier: 1,
  },
  {
    name: "EgoOops",
    venue: "arXiv 2024 / ICCV-W 2025",
    domain: "Multi (circuits, chemistry, crafts)",
    view: "Egocentric",
    hours: 6.8,
    videos: 50,
    stepSegments: "~500",
    hierarchy: "Procedure → Steps",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "Yes (text-required)",
    license: "Research",
    link: "https://github.com/Y-Haneji/EgoOops-annotations",
    notes: "ONLY dataset requiring reference to procedural TEXT to detect mistakes. 6 mistake classes + NL explanations. 5 diverse domains.",
    tier: 1,
  },
  {
    name: "IndustReal",
    venue: "WACV 2024",
    domain: "Industrial assembly",
    view: "Egocentric",
    hours: 5.8,
    videos: 84,
    stepSegments: "~1,000",
    hierarchy: "Procedure → Steps",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "No",
    license: "Research",
    link: "https://github.com/TimSchoonbeek/IndustReal",
    notes: "Closest to manufacturing SOP. 38 error types, 48 valid execution orders. Procedure Step Recognition (PSR) task. Toy-car assembly.",
    tier: 1,
  },
  // Tier 2: Step Annotations + Mistake Detection (No explicit text SOP input)
  {
    name: "Assembly101",
    venue: "CVPR 2022",
    domain: "Toy vehicle assembly",
    view: "Multi-view + Ego",
    hours: 513,
    videos: 4321,
    stepSegments: "1M+",
    hierarchy: "Coarse → Fine actions",
    textSOP: false,
    errorAnnotations: true,
    zeroshottable: "No",
    license: "Research",
    link: "https://assembly-101.github.io/",
    notes: "328 mistake sequences. 3D hand poses. Skill level annotations. Assembly101-O online benchmark used by PREGO.",
    tier: 2,
  },
  {
    name: "EgoPER",
    venue: "CVPR 2024",
    domain: "Kitchen",
    view: "Egocentric",
    hours: 28,
    videos: 385,
    stepSegments: "~4,000",
    hierarchy: "Recipe → Steps",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "No (OCC trained)",
    license: "Research",
    link: "https://www.khoury.northeastern.edu/home/eelhami/egoper.htm",
    notes: "5 error types (omission, correction, modification, slip, addition). One-class classification — trains ONLY on correct videos. RGB+depth+audio+gaze+hand tracking.",
    tier: 2,
  },
  {
    name: "HoloAssist",
    venue: "ICCV 2023",
    domain: "Mixed tasks",
    view: "Egocentric (HoloLens)",
    hours: 166,
    videos: 2221,
    stepSegments: "~20K",
    hierarchy: "Task → Segments",
    textSOP: false,
    errorAnnotations: true,
    zeroshottable: "No",
    license: "Research",
    link: "https://holoassist.github.io/",
    notes: "Instructor-performer interaction. Segment-level mistake labels. Active CVPR 2025 challenge. HoloLens multi-modal (RGB+depth+eye+hand+head).",
    tier: 2,
  },
  {
    name: "Ego4D-PMD",
    venue: "EMNLP 2025",
    domain: "Daily activities",
    view: "Egocentric",
    hours: null,
    videos: null,
    stepSegments: "12,500",
    hierarchy: "Frame + Text",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "Yes (VLM-based)",
    license: "Ego4D License",
    link: "https://openreview.net/forum?id=KkfYxL2L4f",
    notes: "Pairs single video frames with procedural text narrations. Binary mistake labels. Tests VLMs with iterative self-dialog reasoning.",
    tier: 2,
  },
  // Tier 3: Procedural Step Annotations (No error labels)
  {
    name: "EPIC-Kitchens-100",
    venue: "IJCV 2022",
    domain: "Kitchen",
    view: "Egocentric",
    hours: 100,
    videos: 700,
    stepSegments: "90K",
    hierarchy: "Verb + Noun actions",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://epic-kitchens.github.io/2024",
    notes: "97 verb classes, 300 noun classes. Dense temporal annotations. Canonical egocentric benchmark. Action anticipation task.",
    tier: 3,
  },
  {
    name: "COIN",
    venue: "CVPR 2019",
    domain: "180 diverse tasks",
    view: "Third-person (YouTube)",
    hours: 476,
    videos: 11827,
    stepSegments: "46,354",
    hierarchy: "Task → Steps",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Yes (text labels)",
    license: "Research",
    link: "https://coin-dataset.github.io/",
    notes: "Most diverse procedural dataset — car repair, cooking, crafts, etc. 180 tasks. Step temporal boundaries. YouTube sourced.",
    tier: 3,
  },
  {
    name: "CrossTask",
    venue: "CVPR 2019",
    domain: "83 instructional tasks",
    view: "Third-person (YouTube)",
    hours: null,
    videos: 4700,
    stepSegments: "~30K",
    hierarchy: "Task → Ordered Steps",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Yes (step descriptions)",
    license: "Research",
    link: "https://github.com/DmZhukov/CrossTask",
    notes: "Ordered step annotations with text descriptions. Weakly-supervised step localization benchmark. YouTube videos.",
    tier: 3,
  },
  {
    name: "YouCook2",
    venue: "AAAI 2018",
    domain: "Kitchen",
    view: "Third-person (YouTube)",
    hours: 176,
    videos: 2000,
    stepSegments: "15,400",
    hierarchy: "Recipe → Steps",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Yes (step captions)",
    license: "Research",
    link: "http://youcook2.eecs.umich.edu/",
    notes: "89 recipes. Each step has temporal boundary + text description. Gold standard for procedural video captioning.",
    tier: 3,
  },
  {
    name: "Ego-Exo4D",
    venue: "CVPR 2024",
    domain: "Skilled activities (cooking, sports, repair)",
    view: "Ego + Exo synced",
    hours: 740,
    videos: 5035,
    stepSegments: "~17K keysteps",
    hierarchy: "Activity → Keysteps",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Ego4D License",
    link: "https://ego-exo4d-data.org/",
    notes: "Largest multi-view ego-exo dataset. Keystep recognition + skill assessment tasks. Diverse skill levels per scenario.",
    tier: 3,
  },
  {
    name: "EgoExoLearn",
    venue: "CVPR 2024",
    domain: "Kitchen / Repair",
    view: "Ego + Exo async",
    hours: null,
    videos: null,
    stepSegments: "~10K",
    hierarchy: "Activity → Fine actions",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://github.com/OpenGVLab/EgoExoLearn",
    notes: "Bridging ego-exo views. Skill level annotations + gaze signals. 95 verbs, 254 nouns. Left/right hand attribution.",
    tier: 3,
  },
  // Tier 4: Industrial/Manufacturing (Smaller but domain-matched)
  {
    name: "HA-ViD",
    venue: "NeurIPS 2024",
    domain: "Industrial assembly",
    view: "Third-person",
    hours: null,
    videos: null,
    stepSegments: "~3K",
    hierarchy: "Procedure → Steps",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://HA-ViD.github.io/",
    notes: "First human assembly video dataset with representative industrial scenarios. Human-robot shared annotations.",
    tier: 4,
  },
  {
    name: "MECCANO",
    venue: "ECCV 2020",
    domain: "Toy motorcycle assembly",
    view: "Egocentric",
    hours: 7,
    videos: 20,
    stepSegments: "~1,500",
    hierarchy: "Assembly → Actions",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://iplab.dmi.unict.it/MECCANO/",
    notes: "Egocentric hand-object interactions in industrial-like settings. Action recognition + anticipation. Used in ProSkill benchmark.",
    tier: 4,
  },
  {
    name: "IKEA ASM",
    venue: "WACV 2021",
    domain: "Furniture assembly",
    view: "Multi-view + Depth",
    hours: null,
    videos: 371,
    stepSegments: "~8K",
    hierarchy: "Assembly → Atomic actions",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://ikeaasm.github.io/",
    notes: "3 RGB views + depth + pose. 33 action classes. Multi-view temporal action segmentation benchmark.",
    tier: 4,
  },
  {
    name: "IKEA Manuals at Work",
    venue: "NeurIPS 2024 D&B",
    domain: "Furniture assembly",
    view: "Third-person (YouTube)",
    hours: null,
    videos: null,
    stepSegments: "~2K",
    hierarchy: "Manual Steps → Video",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Yes (manual images)",
    license: "Research",
    link: "https://github.com/yunongLiu1/IKEA-Manuals-at-Work",
    notes: "Unique: grounds INSTRUCTION MANUAL images to video. 6DoF part poses + segmentation masks. Closest to 'SOP document → video' paradigm.",
    tier: 4,
  },
  {
    name: "HA4M",
    venue: "Sci Data 2022",
    domain: "Gear train assembly",
    view: "Third-person (Azure Kinect)",
    hours: null,
    videos: null,
    stepSegments: "~500",
    hierarchy: "Assembly → 12 actions",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "CC-BY",
    link: "https://zenodo.org/record/7213301",
    notes: "41 subjects. 6 modalities: RGB + Depth + IR + Aligned + Point Cloud + Skeleton. Lab setting.",
    tier: 4,
  },
  // Tier 5: Surgical Procedures (Richest SOP structure in any domain)
  {
    name: "Cholec80",
    venue: "TMI 2016",
    domain: "Cholecystectomy surgery",
    view: "Endoscopic",
    hours: 40,
    videos: 80,
    stepSegments: "~250K frames",
    hierarchy: "Surgery → 7 Phases",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "CC-BY-NC-SA 4.0",
    link: "https://camma.unistra.fr/datasets/",
    notes: "The 'ImageNet' of surgical phase recognition. 13 surgeons. 25fps. Tool presence annotations at 1fps. Most cited surgical video dataset.",
    tier: 5,
  },
  {
    name: "CholecT50",
    venue: "MIA 2022",
    domain: "Cholecystectomy surgery",
    view: "Endoscopic",
    hours: 28,
    videos: 50,
    stepSegments: "161K triplets",
    hierarchy: "Phase → <Instrument, Verb, Target>",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "CC-BY-NC-SA 4.0",
    link: "https://camma.unistra.fr/datasets/",
    notes: "Fine-grained action triplets. 100 triplet classes. Phase + tool + action annotations. Extends Cholec80.",
    tier: 5,
  },
  {
    name: "MultiBypass140",
    venue: "arXiv 2023",
    domain: "Gastric bypass surgery",
    view: "Endoscopic",
    hours: null,
    videos: 140,
    stepSegments: "Multi-level",
    hierarchy: "Phase → Step → Adverse Events",
    textSOP: true,
    errorAnnotations: true,
    zeroshottable: "No",
    license: "Research",
    link: "https://camma.unistra.fr/datasets/",
    notes: "Multi-centric. Phases + Steps + Intraoperative Adverse Events (IAEs). Error detection in surgery.",
    tier: 5,
  },
  {
    name: "PhaKIR",
    venue: "MICCAI 2024 Challenge",
    domain: "Cholecystectomy surgery",
    view: "Endoscopic",
    hours: null,
    videos: 8,
    stepSegments: "485K frames",
    hierarchy: "Phase → Instruments",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "No",
    license: "Research",
    link: "https://doi.org/10.5281/zenodo.15740619",
    notes: "Multi-institutional (3 centers). Joint phase + keypoint + segmentation. First dataset combining all three tasks.",
    tier: 5,
  },
  // Tier 6: Large-scale unlabeled (for pretraining)
  {
    name: "Egocentric-100K",
    venue: "HuggingFace 2025",
    domain: "Factory (SE Asia)",
    view: "Egocentric",
    hours: 100405,
    videos: 2010759,
    stepSegments: "None",
    hierarchy: "None (unlabeled)",
    textSOP: false,
    errorAnnotations: false,
    zeroshottable: "N/A",
    license: "Apache 2.0",
    link: "https://huggingface.co/datasets/builddotai/Egocentric-100K",
    notes: "Largest manual labor dataset. 14,228 workers. 10.8B frames. SOTA hand visibility. No labels. Factory: assembly, packing, line work.",
    tier: 6,
  },
  {
    name: "HowTo100M",
    venue: "ICCV 2019",
    domain: "Instructional (YouTube)",
    view: "Third-person",
    hours: 134000,
    videos: 1220000,
    stepSegments: "136M clips",
    hierarchy: "Weak (ASR narration)",
    textSOP: true,
    errorAnnotations: false,
    zeroshottable: "Weak supervision",
    license: "Research",
    link: "https://www.di.ens.fr/willow/research/howto100m/",
    notes: "ASR-aligned narrations as weak step labels. 23K different tasks. Foundation for procedural video pretraining. Noisy but massive.",
    tier: 6,
  },
];

const tierLabels = {
  1: "Hierarchical Steps + Errors (Gold Standard)",
  2: "Step Annotations + Mistake Detection",
  3: "Procedural Step Annotations (No Errors)",
  4: "Industrial / Assembly Specific",
  5: "Surgical Procedures",
  6: "Large-Scale Unlabeled / Pretraining",
};

const tierColors = {
  1: "#059669",
  2: "#0284c7",
  3: "#7c3aed",
  4: "#d97706",
  5: "#dc2626",
  6: "#6b7280",
};

const Badge = ({ children, color }) => (
  <span
    style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: "9999px",
      fontSize: "11px",
      fontWeight: 600,
      backgroundColor: color + "18",
      color: color,
      border: `1px solid ${color}30`,
      whiteSpace: "nowrap",
    }}
  >
    {children}
  </span>
);

const BoolBadge = ({ value }) => {
  if (value === true) return <Badge color="#059669">✓ Yes</Badge>;
  if (value === false) return <Badge color="#94a3b8">✗ No</Badge>;
  return <Badge color="#d97706">{value}</Badge>;
};

export default function ProceduralDatasets() {
  const [selectedTier, setSelectedTier] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);

  const filtered = selectedTier
    ? datasets.filter((d) => d.tier === selectedTier)
    : datasets;

  return (
    <div
      style={{
        fontFamily: "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif",
        maxWidth: "100%",
        padding: "24px 16px",
        backgroundColor: "var(--bg, #0f172a)",
        color: "var(--text, #e2e8f0)",
        minHeight: "100vh",
      }}
    >
      <div style={{ marginBottom: 24 }}>
        <h1
          style={{
            fontSize: 22,
            fontWeight: 700,
            margin: 0,
            color: "var(--heading, #f8fafc)",
            letterSpacing: "-0.02em",
          }}
        >
          Procedural Step-Level Video Datasets
        </h1>
        <p
          style={{
            fontSize: 13,
            color: "var(--muted, #94a3b8)",
            margin: "6px 0 16px",
            lineHeight: 1.5,
          }}
        >
          Datasets with temporal step annotations for SOP/procedure verification
          from video — ranked by relevance to zero-shot compliance checking.
          {filtered.length} of {datasets.length} shown.
        </p>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <button
            onClick={() => setSelectedTier(null)}
            style={{
              padding: "5px 12px",
              borderRadius: 6,
              border: `1px solid ${!selectedTier ? "#3b82f6" : "#334155"}`,
              background: !selectedTier ? "#1e3a5f" : "transparent",
              color: !selectedTier ? "#93c5fd" : "#94a3b8",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 500,
            }}
          >
            All ({datasets.length})
          </button>
          {Object.entries(tierLabels).map(([tier, label]) => {
            const count = datasets.filter(
              (d) => d.tier === parseInt(tier)
            ).length;
            const isActive = selectedTier === parseInt(tier);
            return (
              <button
                key={tier}
                onClick={() =>
                  setSelectedTier(isActive ? null : parseInt(tier))
                }
                style={{
                  padding: "5px 12px",
                  borderRadius: 6,
                  border: `1px solid ${isActive ? tierColors[tier] : "#334155"}`,
                  background: isActive ? tierColors[tier] + "20" : "transparent",
                  color: isActive ? tierColors[tier] : "#94a3b8",
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: 500,
                }}
              >
                T{tier}: {label.split("(")[0].trim()} ({count})
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "separate",
            borderSpacing: 0,
            fontSize: 12,
          }}
        >
          <thead>
            <tr>
              {[
                "Dataset",
                "Venue",
                "Domain",
                "View",
                "Hours",
                "Steps",
                "Hierarchy",
                "Text SOP",
                "Errors",
                "Zero-Shot",
              ].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "10px 8px",
                    textAlign: "left",
                    borderBottom: "2px solid #1e293b",
                    color: "#64748b",
                    fontWeight: 600,
                    fontSize: 11,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    position: "sticky",
                    top: 0,
                    background: "var(--bg, #0f172a)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((d, i) => {
              const isExpanded = expandedRow === i;
              return (
                <>
                  <tr
                    key={d.name}
                    onClick={() => setExpandedRow(isExpanded ? null : i)}
                    style={{
                      cursor: "pointer",
                      background: isExpanded
                        ? "#1e293b"
                        : i % 2 === 0
                          ? "transparent"
                          : "#0f172a08",
                      transition: "background 0.15s",
                    }}
                    onMouseEnter={(e) => {
                      if (!isExpanded)
                        e.currentTarget.style.background = "#1e293b50";
                    }}
                    onMouseLeave={(e) => {
                      if (!isExpanded)
                        e.currentTarget.style.background =
                          i % 2 === 0 ? "transparent" : "#0f172a08";
                    }}
                  >
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        fontWeight: 600,
                        color: "#f1f5f9",
                      }}
                    >
                      <div
                        style={{ display: "flex", alignItems: "center", gap: 8 }}
                      >
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background: tierColors[d.tier],
                            flexShrink: 0,
                          }}
                        />
                        <a
                          href={d.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            color: "#93c5fd",
                            textDecoration: "none",
                          }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {d.name}
                        </a>
                      </div>
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#94a3b8",
                        fontSize: 11,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.venue}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#cbd5e1",
                        maxWidth: 140,
                      }}
                    >
                      {d.domain}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#94a3b8",
                        fontSize: 11,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.view}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#e2e8f0",
                        fontVariantNumeric: "tabular-nums",
                        textAlign: "right",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.hours ? d.hours.toLocaleString() + "h" : "—"}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#e2e8f0",
                        textAlign: "right",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.stepSegments}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        color: "#94a3b8",
                        fontSize: 11,
                      }}
                    >
                      {d.hierarchy}
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        textAlign: "center",
                      }}
                    >
                      <BoolBadge value={d.textSOP} />
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        textAlign: "center",
                      }}
                    >
                      <BoolBadge value={d.errorAnnotations} />
                    </td>
                    <td
                      style={{
                        padding: "10px 8px",
                        borderBottom: "1px solid #1e293b",
                        textAlign: "center",
                      }}
                    >
                      <BoolBadge value={d.zeroshottable} />
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr key={d.name + "-detail"}>
                      <td
                        colSpan={10}
                        style={{
                          padding: "12px 16px 16px 32px",
                          borderBottom: "1px solid #1e293b",
                          background: "#1e293b",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 12,
                            lineHeight: 1.6,
                            color: "#cbd5e1",
                          }}
                        >
                          <strong style={{ color: "#f1f5f9" }}>Notes:</strong>{" "}
                          {d.notes}
                          <br />
                          <strong style={{ color: "#f1f5f9" }}>
                            License:
                          </strong>{" "}
                          {d.license}
                          {d.videos && (
                            <>
                              {" "}
                              &nbsp;·&nbsp;{" "}
                              <strong style={{ color: "#f1f5f9" }}>
                                Videos:
                              </strong>{" "}
                              {d.videos.toLocaleString()}
                            </>
                          )}
                          <br />
                          <a
                            href={d.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: "#60a5fa", fontSize: 11 }}
                          >
                            {d.link} ↗
                          </a>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>

      <div
        style={{
          marginTop: 24,
          padding: 16,
          background: "#1e293b",
          borderRadius: 8,
          fontSize: 12,
          lineHeight: 1.7,
          color: "#94a3b8",
        }}
      >
        <strong style={{ color: "#f1f5f9" }}>Key takeaway:</strong> Only 4
        datasets (Tier 1) combine text-described procedures + video + step-level
        temporal annotations + error labels. Of those, only{" "}
        <strong style={{ color: "#60a5fa" }}>EgoOops</strong> requires the model
        to reference procedural text to detect mistakes, and only{" "}
        <strong style={{ color: "#60a5fa" }}>IndustReal</strong> targets
        industrial settings. No dataset provides free-text SOPs as input for
        zero-shot compliance verification — this is the gap a{" "}
        <strong style={{ color: "#f1f5f9" }}>SOPBench</strong> would fill.
        <br />
        <span style={{ fontSize: 11, color: "#64748b" }}>
          Click any row to expand details. Dot color = tier. Data compiled from
          NeurIPS, CVPR, ICCV, WACV, EMNLP 2022–2026 publications.
        </span>
      </div>
    </div>
  );
}

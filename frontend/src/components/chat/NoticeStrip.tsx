// GO-9/10/11 notices and the private-upload processing notice. The emergency notice is rendered
// above the answer (see AnswerBubble); everything else goes in the strip under it.
import { NOTICE_LABEL } from "@/lib/labels";
import { isArabicScript } from "@/lib/text";
import type { Notice, NoticeCode } from "@/lib/types";

import { cx } from "../ui";

const ICON: Record<NoticeCode, string> = {
  emergency: "!",
  not_advice: "i",
  referral: "→",
  dated_information: "⏱",
  third_party_processing: "⇄",
};

const TONE: Record<NoticeCode, string> = {
  emergency: "border-bad/50 bg-bad/10",
  not_advice: "border-info/40 bg-info/10",
  referral: "border-info/40 bg-info/10",
  dated_information: "border-warn/40 bg-warn/10",
  third_party_processing: "border-mine/40 bg-mine/10",
};

const LABEL_TONE: Record<NoticeCode, string> = {
  emergency: "text-bad",
  not_advice: "text-info",
  referral: "text-info",
  dated_information: "text-warn",
  third_party_processing: "text-mine",
};

export function NoticeItem({ notice }: { notice: Notice }) {
  const urdu = isArabicScript(notice.message);
  return (
    <li
      role={notice.code === "emergency" ? "alert" : undefined}
      className={cx("flex gap-2 rounded-md border px-3 py-2 text-sm", TONE[notice.code])}
    >
      <span
        aria-hidden="true"
        className={cx(
          "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-current text-xs font-bold",
          LABEL_TONE[notice.code],
        )}
      >
        {ICON[notice.code]}
      </span>
      <div className="min-w-0">
        <p className={cx("text-xs font-semibold uppercase tracking-wide", LABEL_TONE[notice.code])}>
          {NOTICE_LABEL[notice.code]}
        </p>
        <p dir="auto" lang={urdu ? "ur" : undefined} className="text-fg">
          {notice.message}
        </p>
        {notice.contact && (
          <p dir="auto" className="mt-0.5 text-fg">
            <span className="font-medium">Contact:</span> {notice.contact}
          </p>
        )}
        {notice.source_url && (
          <a
            href={notice.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-0.5 inline-block break-all font-medium text-accent underline underline-offset-2"
          >
            Check the official source<span className="sr-only"> (opens in a new tab)</span> ↗
          </a>
        )}
      </div>
    </li>
  );
}

export function NoticeStrip({ notices, label }: { notices: Notice[]; label: string }) {
  if (notices.length === 0) return null;
  return (
    <ul aria-label={label} className="flex flex-col gap-2">
      {notices.map((n, i) => (
        <NoticeItem key={`${n.code}-${i}`} notice={n} />
      ))}
    </ul>
  );
}

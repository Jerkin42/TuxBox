//! LOCAL: time spent per named span (`TUXBOX_TIMING=1`), logged every 10 s.
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

type Table = HashMap<&'static str, (u32, Duration, Duration)>;
static SPANS: LazyLock<Mutex<(Table, Instant)>> = LazyLock::new(|| Mutex::new((Table::new(), Instant::now())));

pub(crate) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TUXBOX_TIMING").is_some());
    *ON
}

pub(crate) fn add(name: &'static str, spent: Duration) {
    let mut guard = SPANS.lock().unwrap_or_else(|e| e.into_inner());
    let entry = guard.0.entry(name).or_default();
    *entry = (entry.0 + 1, entry.1 + spent, entry.2.max(spent));
    let wall = guard.1.elapsed();
    if wall < Duration::from_secs(10) { return; }
    let mut rows: Vec<_> = guard.0.drain().collect();
    guard.1 = Instant::now();
    drop(guard);
    rows.sort_by_key(|r| std::cmp::Reverse(r.1.1));
    let text = rows.iter().take(26).map(|(n, (c, t, m))| format!("{n}={}ms/{c}(max {})", t.as_millis(), m.as_millis())).collect::<Vec<_>>();
    for part in text.chunks(9) {
        crate::diagnostics::log(&format!("TIMING {:.1}s {}", wall.as_secs_f32(), part.join(" ")));
    }
}

pub(crate) struct Span(&'static str, Option<Instant>);
pub(crate) fn span(name: &'static str) -> Span { Span(name, on().then(Instant::now)) }
impl Drop for Span {
    fn drop(&mut self) { if let Some(start) = self.1 { add(self.0, start.elapsed()); } }
}

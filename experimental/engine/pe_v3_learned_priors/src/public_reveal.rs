//! Causal public-information metadata for determinized search states.
//!
//! The completed `Pokemon` values in a determinized state are simulator truth,
//! not necessarily player knowledge.  This mask is therefore the only source
//! of truth for information features used below the search root.

use crate::choices::Choices;
use crate::engine::abilities::Abilities;
use crate::engine::items::Items;
use crate::instruction::{Instruction, PublicRevealInstruction, StateInstructions};
use crate::state::{LastUsedMove, PokemonIndex, PokemonMoveIndex, Side, SideReference, State};

const SPECIES_OFFSET: u32 = 0;
const MOVES_OFFSET: u32 = 6;
const ITEMS_OFFSET: u32 = 30;
const ABILITIES_OFFSET: u32 = 36;
const VALID_BITS: u64 = (1_u64 << 42) - 1;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PublicRevealMask(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevealField {
    Species,
    Move(PokemonMoveIndex),
    Item,
    Ability,
}

impl PublicRevealMask {
    pub fn from_bits(bits: u64) -> Self {
        Self(bits & VALID_BITS)
    }

    pub fn bits(self) -> u64 {
        self.0
    }

    pub fn from_public_side(side: &Side) -> Self {
        let mut mask = Self::default();
        let pokemon_indices = [
            PokemonIndex::P0,
            PokemonIndex::P1,
            PokemonIndex::P2,
            PokemonIndex::P3,
            PokemonIndex::P4,
            PokemonIndex::P5,
        ];
        let move_indices = [
            PokemonMoveIndex::M0,
            PokemonMoveIndex::M1,
            PokemonMoveIndex::M2,
            PokemonMoveIndex::M3,
        ];
        for (pokemon_offset, pokemon) in side.pokemon.into_iter().enumerate() {
            let pokemon_index = pokemon_indices[pokemon_offset];
            if pokemon.id.to_string().eq_ignore_ascii_case("none") {
                continue;
            }
            mask.reveal(pokemon_index, RevealField::Species);
            for (move_offset, pokemon_move) in pokemon.moves.into_iter().enumerate() {
                if pokemon_move.id != Choices::NONE {
                    mask.reveal(pokemon_index, RevealField::Move(move_indices[move_offset]));
                }
            }
            if pokemon.item != Items::NONE && pokemon.item != Items::UNKNOWNITEM {
                mask.reveal(pokemon_index, RevealField::Item);
            }
            if pokemon.ability != Abilities::NONE {
                mask.reveal(pokemon_index, RevealField::Ability);
            }
        }
        mask
    }

    fn bit(pokemon: PokemonIndex, field: RevealField) -> u64 {
        let pokemon = pokemon as u32;
        let offset = match field {
            RevealField::Species => SPECIES_OFFSET + pokemon,
            RevealField::Move(move_index) => MOVES_OFFSET + pokemon * 4 + move_index as u32,
            RevealField::Item => ITEMS_OFFSET + pokemon,
            RevealField::Ability => ABILITIES_OFFSET + pokemon,
        };
        1_u64 << offset
    }

    pub fn contains(self, pokemon: PokemonIndex, field: RevealField) -> bool {
        self.0 & Self::bit(pokemon, field) != 0
    }

    pub fn reveal(&mut self, pokemon: PokemonIndex, field: RevealField) {
        self.0 |= Self::bit(pokemon, field);
    }

    pub fn hide(&mut self, pokemon: PokemonIndex, field: RevealField) {
        self.0 &= !Self::bit(pokemon, field);
    }

    pub fn species_fraction(self) -> f32 {
        (self.0 & 0x3f).count_ones() as f32 / 6.0
    }

    pub fn moves_fraction(self) -> f32 {
        ((self.0 >> MOVES_OFFSET) & 0x00ff_ffff).count_ones() as f32 / 24.0
    }

    pub fn items_fraction(self) -> f32 {
        ((self.0 >> ITEMS_OFFSET) & 0x3f).count_ones() as f32 / 6.0
    }

    pub fn abilities_fraction(self) -> f32 {
        ((self.0 >> ABILITIES_OFFSET) & 0x3f).count_ones() as f32 / 6.0
    }
}

impl State {
    /// Return what `observer` knows about the opposing side.
    pub fn public_reveals(&self, observer: SideReference) -> PublicRevealMask {
        match observer {
            SideReference::SideOne => self.s1_public_reveals,
            SideReference::SideTwo => self.s2_public_reveals,
        }
    }

    pub fn set_public_reveals(&mut self, observer: SideReference, mask: PublicRevealMask) {
        match observer {
            SideReference::SideOne => self.s1_public_reveals = mask,
            SideReference::SideTwo => self.s2_public_reveals = mask,
        }
    }

    pub fn initialize_public_reveals(&mut self, observer: SideReference) {
        let subject = observer.get_other_side();
        let mask = PublicRevealMask::from_public_side(self.get_side_immutable(&subject));
        self.set_public_reveals(observer, mask);
    }

    pub fn reveal_to_opponent(
        &mut self,
        subject: SideReference,
        pokemon: PokemonIndex,
        field: RevealField,
    ) {
        let observer = subject.get_other_side();
        let mut mask = self.public_reveals(observer);
        mask.reveal(pokemon, field);
        self.set_public_reveals(observer, mask);
    }

    pub fn hide_from_opponent(
        &mut self,
        subject: SideReference,
        pokemon: PokemonIndex,
        field: RevealField,
    ) {
        let observer = subject.get_other_side();
        let mut mask = self.public_reveals(observer);
        mask.hide(pokemon, field);
        self.set_public_reveals(observer, mask);
    }
}

fn mask_for_observer(
    s1: PublicRevealMask,
    s2: PublicRevealMask,
    subject: SideReference,
) -> PublicRevealMask {
    match subject {
        SideReference::SideOne => s2,
        SideReference::SideTwo => s1,
    }
}

fn mask_for_observer_mut<'a>(
    s1: &'a mut PublicRevealMask,
    s2: &'a mut PublicRevealMask,
    subject: SideReference,
) -> &'a mut PublicRevealMask {
    match subject {
        SideReference::SideOne => s2,
        SideReference::SideTwo => s1,
    }
}

/// Append reversible reveal deltas to every stochastic outcome.
///
/// This scans the public instruction tape, never the completed hidden fields.
/// A delta is emitted only when the root mask did not already contain the fact,
/// which makes ordinary reverse-instruction traversal exact and sibling-safe.
pub fn append_causal_reveal_instructions(state: &State, outcomes: &mut [StateInstructions]) {
    for outcome in outcomes {
        let mut s1_mask = state.s1_public_reveals;
        let mut s2_mask = state.s2_public_reveals;
        let mut active = [state.side_one.active_index, state.side_two.active_index];
        let mut additions = Vec::new();

        let mut reveal = |subject: SideReference,
                          pokemon_index: PokemonIndex,
                          field: RevealField| {
            let current = mask_for_observer(s1_mask, s2_mask, subject);
            if current.contains(pokemon_index, field) {
                return;
            }
            mask_for_observer_mut(&mut s1_mask, &mut s2_mask, subject).reveal(pokemon_index, field);
            additions.push(Instruction::PublicReveal(PublicRevealInstruction {
                subject,
                pokemon_index,
                field,
            }));
        };

        for instruction in &outcome.instruction_list {
            match instruction {
                Instruction::RecordAction(record) => reveal(
                    record.side_ref,
                    record.pokemon_index,
                    RevealField::Move(record.move_index),
                ),
                Instruction::SetLastUsedMove(record) => {
                    if let LastUsedMove::Move(move_index) = record.last_used_move {
                        let side_offset = match record.side_ref {
                            SideReference::SideOne => 0,
                            SideReference::SideTwo => 1,
                        };
                        reveal(
                            record.side_ref,
                            active[side_offset],
                            RevealField::Move(move_index),
                        );
                    }
                }
                Instruction::Switch(switch) => {
                    reveal(switch.side_ref, switch.next_index, RevealField::Species);
                    let side_offset = match switch.side_ref {
                        SideReference::SideOne => 0,
                        SideReference::SideTwo => 1,
                    };
                    active[side_offset] = switch.next_index;
                }
                Instruction::RecordItemActivation(record) => {
                    reveal(record.side_ref, record.pokemon_index, RevealField::Item)
                }
                Instruction::ChangeItem(change) => {
                    let side_offset = match change.side_ref {
                        SideReference::SideOne => 0,
                        SideReference::SideTwo => 1,
                    };
                    reveal(change.side_ref, active[side_offset], RevealField::Item);
                }
                Instruction::ChangeAbility(change) => {
                    let side_offset = match change.side_ref {
                        SideReference::SideOne => 0,
                        SideReference::SideTwo => 1,
                    };
                    reveal(change.side_ref, active[side_offset], RevealField::Ability);
                }
                Instruction::PublicReveal(existing) => {
                    reveal(existing.subject, existing.pokemon_index, existing.field)
                }
                _ => {}
            }
        }
        outcome.instruction_list.extend(additions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::{
        ChangeAbilityInstruction, ChangeItemInstruction, SetLastUsedMoveInstruction,
        SwitchInstruction,
    };

    #[test]
    fn packed_mask_counts_each_public_field_without_overlap() {
        let mut mask = PublicRevealMask::default();
        mask.reveal(PokemonIndex::P5, RevealField::Species);
        mask.reveal(PokemonIndex::P5, RevealField::Move(PokemonMoveIndex::M3));
        mask.reveal(PokemonIndex::P5, RevealField::Item);
        mask.reveal(PokemonIndex::P5, RevealField::Ability);

        assert_eq!(mask.species_fraction(), 1.0 / 6.0);
        assert_eq!(mask.moves_fraction(), 1.0 / 24.0);
        assert_eq!(mask.items_fraction(), 1.0 / 6.0);
        assert_eq!(mask.abilities_fraction(), 1.0 / 6.0);
        assert_eq!(PublicRevealMask::from_bits(mask.bits()), mask);
    }

    #[test]
    fn invalid_serialized_bits_are_discarded() {
        assert_eq!(PublicRevealMask::from_bits(u64::MAX).bits(), VALID_BITS);
    }

    #[test]
    fn branch_reveals_are_causal_reversible_and_do_not_clear_root_facts() {
        let mut state = State::default();
        state
            .s1_public_reveals
            .reveal(PokemonIndex::P1, RevealField::Species);
        let root_mask = state.s1_public_reveals;
        let mut outcomes = vec![StateInstructions {
            percentage: 100.0,
            instruction_list: vec![
                Instruction::SetLastUsedMove(SetLastUsedMoveInstruction {
                    side_ref: SideReference::SideTwo,
                    last_used_move: LastUsedMove::Move(PokemonMoveIndex::M2),
                    previous_last_used_move: LastUsedMove::None,
                }),
                Instruction::Switch(SwitchInstruction {
                    side_ref: SideReference::SideTwo,
                    previous_index: PokemonIndex::P0,
                    next_index: PokemonIndex::P1,
                }),
                Instruction::ChangeItem(ChangeItemInstruction {
                    side_ref: SideReference::SideTwo,
                    current_item: Items::NONE,
                    new_item: Items::LEFTOVERS,
                }),
                Instruction::ChangeAbility(ChangeAbilityInstruction {
                    side_ref: SideReference::SideTwo,
                    ability_change: 1,
                }),
            ],
        }];

        append_causal_reveal_instructions(&state, &mut outcomes);
        let reveal_count = outcomes[0]
            .instruction_list
            .iter()
            .filter(|instruction| matches!(instruction, Instruction::PublicReveal(_)))
            .count();
        // P1 species was public at the root, so only move/item/ability are deltas.
        assert_eq!(reveal_count, 3);

        state.apply_instructions(&outcomes[0].instruction_list);
        assert!(state
            .s1_public_reveals
            .contains(PokemonIndex::P0, RevealField::Move(PokemonMoveIndex::M2),));
        assert!(state
            .s1_public_reveals
            .contains(PokemonIndex::P1, RevealField::Item));
        assert!(state
            .s1_public_reveals
            .contains(PokemonIndex::P1, RevealField::Ability));

        state.reverse_instructions(&outcomes[0].instruction_list);
        assert_eq!(state.s1_public_reveals, root_mask);
    }

    #[test]
    fn state_serialization_round_trips_masks_and_reads_legacy_states() {
        let mut state = State::default();
        state
            .s1_public_reveals
            .reveal(PokemonIndex::P3, RevealField::Move(PokemonMoveIndex::M1));
        state
            .s2_public_reveals
            .reveal(PokemonIndex::P2, RevealField::Item);
        let serialized = state.serialize();
        let restored = State::deserialize(&serialized);
        assert_eq!(restored.s1_public_reveals, state.s1_public_reveals);
        assert_eq!(restored.s2_public_reveals, state.s2_public_reveals);

        let legacy = serialized.split('/').take(11).collect::<Vec<_>>().join("/");
        let restored_legacy = State::deserialize(&legacy);
        assert_eq!(restored_legacy.s1_public_reveals.bits(), 0);
        assert_eq!(restored_legacy.s2_public_reveals.bits(), 0);
    }
}
